#!/bin/bash
cd ..

# List of models to train
models=("resnet18" "resnet34" "resnet50" "vgg16" "alexnet" "mobilenetv3")

# Loop through each model and train
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Training model: $model"
    echo "=========================================="
    python train.py --config config/mit.json --model_name "$model"
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed training for $model"
    else
        echo "✗ Training failed for $model"
    fi
    echo ""
done

echo "All training jobs completed!"

