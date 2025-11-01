#!/bin/bash
cd ..

# List of models to train
models=("resnet18" "resnet34" "resnet50" "vgg16" "alexnet" "mobilenetv3")

# Loop through each model and train
for model in "${models[@]}"; do
    for optimizer in "adam" "sgd" "adamw"; do
        echo "=========================================="
        echo "Training model: $model with optimizer: $optimizer"
        echo "=========================================="
        python train.py --config config/${model}_${optimizer}.json --dataset caltech101 --input_size 224 --batch_size 32

        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed training for $model with optimizer: $optimizer"
        else
            echo "✗ Training failed for $model with optimizer: $optimizer"
        fi
        echo ""
    done
done

echo "All training jobs completed!"

