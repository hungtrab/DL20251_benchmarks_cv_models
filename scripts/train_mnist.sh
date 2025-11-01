#!/bin/bash
cd ..

# List of models to train
models=("lenet" "alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "resnet101" "inceptionv3" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for optimizer in "adam" "sgd" "adamw"; do
        echo "=========================================="
        echo "Training model: $model with optimizer: $optimizer"
        echo "=========================================="
        python train.py --config config/${model}_${optimizer}.json --dataset mnist --input_size 28 --batch_size 64
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed training for $model with optimizer: $optimizer"
        else
            echo "✗ Training failed for $model with optimizer: $optimizer"
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

