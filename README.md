# Code for paper A Framework for Neural Topic Modeling with Mutual Information and Group Regularization

In this source code, we have three cases of applying NeuroMig to ECRTM, ETM, FASTopic. In the future, we will reorganize the source code to only one big source code for almost neural topic models that can apply NeuroMig. For each folder of each framework in the source code, we can run the code by the follwing steps.

## Preparing libraries
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar for evaluating
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.
5. I need to have the wandb api key to track the metrics of the method.

## Usage
To run and evaluate our model for 20NG dataset, cd to each of the folder and run this example:

> python main.py --model NeuroMig --dataset 20NG --num_topics 50 --beta_temp 0.2 --num_groups 20 --epochs 500 --device cuda --use_pretrainWE --weight_ECR "$weight_ECR" --weight_GR "$weight_GR" --weight_InfoNCE "$weight_InfoNCE"

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.

