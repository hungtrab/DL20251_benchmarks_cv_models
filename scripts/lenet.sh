#!/bin/bash
cd ..

# List of models to train
models=("lenet")

# Loop through each model and train
for model in "${models[@]}"; do
    for scheduler in "constant" "cosine"; do
        for dataset in "fashionmnist"; do
            echo "=========================================="
            echo "Training model: $model with scheduler: $scheduler on dataset: $dataset"
            echo "=========================================="
            python train.py --config config/${model}_adamw.json --dataset $dataset --input_size 28 --batch_size 128 --num_warmup_steps 100 --use_tensorboard --tensorboard_log_dir results/tensorboard/${model}_${scheduler}_${dataset} --use_wandb --wandb_project dl20251-cv --wandb_run_name ${model}_${scheduler}_${dataset} --scheduler $scheduler --num_epochs 20
            
            # Check if training was successful
            if [ $? -eq 0 ]; then
                echo "✓ Successfully completed training for $model with scheduler: $scheduler on dataset: $dataset"
            else
                echo "✗ Training failed for $model with scheduler: $scheduler on dataset: $dataset"
            fi
            echo ""
        done
    done
done

echo "All training jobs completed!"

