#!/bin/bash
cd ..

# List of models to train
models=("lenet" "alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3_s" "mobilenetv3_l" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "resnet101" "inceptionv3" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for scheduler in "constant" "cosine"; do
        echo "=========================================="
        echo "Training model: $model with scheduler: $scheduler"
        echo "=========================================="
        python train.py --config config/${model}_adamw.json --dataset mnist --input_size 28 --batch_size 64 --use_tensorboard --tensorboard_log_dir results/tensorboard/${model}_${scheduler} --use_wandb --wandb_project dl20251-cv --wandb_run_name ${model}_${scheduler}_mnist --scheduler $scheduler
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed training for $model with scheduler: $scheduler"
        else
            echo "✗ Training failed for $model with scheduler: $scheduler"
        fi
        echo ""
    done
done

#     echo "=========================================="
#     echo "Training model: $model"
#     echo "=========================================="
#     python train.py --config config/${model}.json --dataset mnist
    
#     # Check if training was successful
#     if [ $? -eq 0 ]; then
#         echo "✓ Successfully completed training for $model"
#     else
#         echo "✗ Training failed for $model"
#     fi
#     echo ""

echo "All training jobs completed!"

