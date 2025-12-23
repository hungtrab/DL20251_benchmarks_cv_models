#!/bin/bash
cd ..

# List of models to train
models=("alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3_s" "mobilenetv3_l" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for scheduler in "constant" "cosine"; do
        echo "=========================================="
        echo "Training model: $model with scheduler: $scheduler"
        echo "=========================================="
        python train.py --config config/${model}_adamw.json --dataset cifar100_224 --input_size 224 --batch_size 128 --num_warmup_steps 500 --use_tensorboard --tensorboard_log_dir results/tensorboard/${model}_${scheduler}_cifar100_224 --use_wandb --wandb_project dl20251-cv --wandb_run_name ${model}_${scheduler}_cifar100_224 --scheduler $scheduler --num_epochs 55
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed training for $model with scheduler: $scheduler"
        else:
            echo "✗ Training failed for $model with scheduler: $scheduler"
        fi
        echo ""
    done
done

echo "All training jobs completed!"

