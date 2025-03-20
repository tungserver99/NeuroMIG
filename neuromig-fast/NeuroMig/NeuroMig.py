import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from .GR import GR
from ._ETP import ETP
import torch_kmeans
import logging
import sentence_transformers

def pairwise_euclidean_distance(x, y):
    cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    return cost

class SimpleSentenceEmbeddingNN(nn.Module):
    def __init__(self, input_dim=5000, hidden_dim_1=384, hidden_dim_2=384, embedding_dim=384, dropout_prob=0.2):
        super(SimpleSentenceEmbeddingNN, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.layer3 = nn.Linear(hidden_dim_2, embedding_dim)

        # Optional components
        self.activation = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim_1)
        self.norm2 = nn.LayerNorm(hidden_dim_2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Pass input through the first layer
        x = self.layer1(x)
        x = self.activation(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # Pass through the second layer
        x = self.layer2(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # Pass through the third layer to produce embeddings
        x = self.layer3(x)

        # Optionally normalize embeddings to a unit sphere
        x = F.normalize(x, p=2, dim=1)

        return x


class NeuroMig(nn.Module):
    def __init__(self, vocab_size, num_topics=50, num_groups=10, 
                 hidden_dim_1=384, hidden_dim_2=384, embed_size=384, 
                 dropout=0., beta_temp=0.2, theta_temp: float=1.0,
                 DT_alpha: float=3.0, TW_alpha: float=2.0,
                 weight_loss_GR=250.0,
                 alpha_GR=20.0, sinkhorn_max_iter=1000,
                 weight_loss_InfoNCE=10.0):
        super().__init__()

        self.num_topics = num_topics
        self.num_groups = num_groups
        self.beta_temp = beta_temp
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp

        self.epsilon = 1e-12
        
        self.simpleembedding = SimpleSentenceEmbeddingNN(input_dim=vocab_size, hidden_dim_1=embed_size, hidden_dim_2=embed_size, embedding_dim=embed_size, dropout_prob=dropout)
        
        self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

        self.num_topics_per_group = num_topics // num_groups
        self.GR = GR(weight_loss_GR, alpha_GR, sinkhorn_max_iter)
        self.group_connection_regularizer = None

        # for InfoNCE
        self.prj_bert = nn.Sequential()
        self.weight_loss_InfoNCE = weight_loss_InfoNCE

    def create_group_connection_regularizer(self):
        kmean_model = torch_kmeans.KMeans(
            n_clusters=self.num_groups, max_iter=1000, seed=0, verbose=False,
            normalize='unit')
        group_id = kmean_model.fit_predict(self.topic_embeddings.reshape(
            1, self.topic_embeddings.shape[0], self.topic_embeddings.shape[1]))
        group_id = group_id.reshape(-1)
        self.group_topic = [[] for _ in range(self.num_groups)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i)

        self.group_connection_regularizer = torch.ones(
            (self.num_topics, self.num_topics), device=self.topic_embeddings.device) / 5.
        for i in range(self.num_topics):
            for j in range(self.num_topics):
                if group_id[i] == group_id[j]:
                    self.group_connection_regularizer[i][j] = 1
        self.group_connection_regularizer.fill_diagonal_(0)
        self.group_connection_regularizer = self.group_connection_regularizer.clamp(min=1e-4)
        for _ in range(50):
            self.group_connection_regularizer = self.group_connection_regularizer / \
                self.group_connection_regularizer.sum(axis=1, keepdim=True) / self.num_topics
            self.group_connection_regularizer = (self.group_connection_regularizer \
                + self.group_connection_regularizer.T) / 2.

    def get_transp_DT(self,
                      doc_embeddings,
                    ):

        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()
    
    # only for testing
    def get_beta(self):
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]

        return beta

    # only for testing
    def get_theta(self,
                  doc_simple_embeddings,
                  train_doc_simple_embeddings
                ):
        # doc_embeddings = self.simpleembedding(bow)
        topic_embeddings = self.topic_embeddings.detach().to(doc_simple_embeddings.device)
        dist = pairwise_euclidean_distance(doc_simple_embeddings, topic_embeddings)
        train_dist = pairwise_euclidean_distance(train_doc_simple_embeddings, topic_embeddings)

        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)

        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)

        return theta

    def sim(self, rep, bert):
        prep = self.prj_bert(rep)
        pbert = self.prj_bert(bert)
        return torch.exp(F.cosine_similarity(prep, pbert))

    def csim(self, rep, bert):
        pbow = self.prj_bert(rep)
        pbert = self.prj_bert(bert)
        csim_matrix = (pbow@pbert.T) / (pbow.norm(keepdim=True,
                                                  dim=-1)@pbert.norm(keepdim=True, dim=-1).T)
        csim_matrix = torch.exp(csim_matrix)
        csim_matrix = csim_matrix / csim_matrix.sum(dim=1, keepdim=True)
        return -csim_matrix.log()

    def compute_loss_InfoNCE(self, rep, contextual_emb):
        if self.weight_loss_InfoNCE <= 1e-6:
            return 0.
        else:
            sim_matrix = self.csim(rep, contextual_emb)
            return sim_matrix.diag().mean() * self.weight_loss_InfoNCE

    def get_loss_GR(self):
        cost = pairwise_euclidean_distance(
            self.topic_embeddings, self.topic_embeddings) + 1e1 * torch.ones(self.num_topics, self.num_topics).to(self.topic_embeddings.device)
        loss_GR = self.GR(cost, self.group_connection_regularizer)
        return loss_GR

    def forward(self, input, epoch_id=None):
        bow = input["data"]
        contextual_emb = input["contextual_embed"]
        doc_simple_embedding = self.simpleembedding(bow)

        loss_DT, transp_DT = self.DT_ETP(doc_simple_embedding, self.topic_embeddings)
        #loss_DT, transp_DT = self.DT_ETP(contextual_emb, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]

        # Dual Semantic-relation Reconstruction
        recon = torch.matmul(theta, beta)

        loss_DSR = -(bow * (recon + self.epsilon).log()).sum(axis=1).mean()

        loss_TM = loss_DSR + loss_ETP

        loss_InfoNCE = self.compute_loss_InfoNCE(doc_simple_embedding, contextual_emb)
        if epoch_id == 5 and self.group_connection_regularizer is None:
            self.create_group_connection_regularizer()
        if self.group_connection_regularizer is not None and epoch_id > 5:
            loss_GR = self.get_loss_GR()
        else:
            loss_GR = 0.

        loss = loss_TM + loss_GR + loss_InfoNCE

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_GR': loss_GR,
            'loss_InfoNCE': loss_InfoNCE,
        }
        # rst_dict = {
        #     'loss': loss_TM
        # }

        return rst_dict

# class NeuroMig(nn.Module):
#     def __init__(self, vocab_size, num_topics=50, num_groups=10, 
#                  hidden_dim_1=384, hidden_dim_2=384, embed_size=384, 
#                  dropout=0., beta_temp=0.2, theta_temp: float=1.0,
#                  DT_alpha: float=3.0, TW_alpha: float=2.0,
#                  weight_loss_GR=250.0,
#                  alpha_GR=20.0, sinkhorn_max_iter=1000,
#                  weight_loss_InfoNCE=10.0):
#         super().__init__()

#         self.num_topics = num_topics
#         self.num_groups = num_groups
#         self.beta_temp = beta_temp
#         self.DT_alpha = DT_alpha
#         self.TW_alpha = TW_alpha
#         self.theta_temp = theta_temp

#         self.epsilon = 1e-12
        
#         # self.simpleembedding = SimpleSentenceEmbeddingNN(input_dim=vocab_size, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, embedding_dim=embed_size, dropout_prob=dropout)
        
#         self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
#         self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

#         self.topic_embeddings = torch.empty((self.num_topics, embed_size))
#         nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
#         self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

#         self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
#         self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))

#         self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
#         self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

#     def get_transp_DT(self,
#                       doc_embeddings,
#                     ):

#         topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
#         _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

#         return transp.detach().cpu().numpy()

#     # only for testing
#     def get_beta(self):
#         _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
#         # use transport plan as beta
#         beta = transp_TW * transp_TW.shape[0]

#         return beta

#     # only for testing
#     def get_theta(self,
#                   doc_embeddings,
#                   train_doc_embeddings
#                 ):
#         topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
#         dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
#         train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

#         exp_dist = torch.exp(-dist / self.theta_temp)
#         exp_train_dist = torch.exp(-train_dist / self.theta_temp)

#         theta = exp_dist / (exp_train_dist.sum(0))
#         theta = theta / theta.sum(1, keepdim=True)

#         return theta

#     def forward(self, train_bow, doc_embeddings):
#         loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
#         loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

#         loss_ETP = loss_DT + loss_TW

#         theta = transp_DT * transp_DT.shape[0]
#         beta = transp_TW * transp_TW.shape[0]

#         # Dual Semantic-relation Reconstruction
#         recon = torch.matmul(theta, beta)

#         loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()

#         loss = loss_DSR + loss_ETP

#         rst_dict = {
#             'loss': loss,
#         }

#         return rst_dict



