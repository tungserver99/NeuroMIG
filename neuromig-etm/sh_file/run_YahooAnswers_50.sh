#!/bin/bash

# Define the arrays for weight_GR and weight_InfoNCE
weight_GR_values=(1 5 10 20 50)
weight_InfoNCE_values=(1 10 30 50 80 100 130 150)

# Loop through each combination of weight_GR and weight_InfoNCE
for weight_GR in "${weight_GR_values[@]}"; do
    for weight_InfoNCE in "${weight_InfoNCE_values[@]}"; do
        # Run the python command with the current combination of weight_GR and weight_InfoNCE
        python main.py --wandb_prj neuromax_Yahoo_50K --model NeuroMax --dataset YahooAnswers --num_topics 50 --beta_temp 0.2 --num_groups 20 --epochs 500 --device cuda --lr 0.002 --lr_scheduler StepLR --dropout 0.2 --batch_size 200 --lr_step_size 125 --use_pretrainWE --weight_ECR 40 --weight_GR "$weight_GR" --alpha_ECR 20.0 --alpha_GR 5.0 --weight_InfoNCE "$weight_InfoNCE"
    done
done
