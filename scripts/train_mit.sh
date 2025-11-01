#!/bin/bash

# Check if intel_image directory exists
cd ..

if [ ! -d "data/mit_indoor" ]; then
    echo "MIT indoor dataset not found. Downloading..."
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    # Install gdown if not already installed
    pip install -q gdown
    
    # Download the dataset
    gdown https://drive.google.com/uc?id=17aWl6kKKkgEmt1HmiFkJUZRomRWjApwg -O data/mit_indoor.zip

    # Unzip the dataset
    echo "Extracting dataset..."
    unzip -q data/mit_indoor.zip -d data/mit_indoor

    # Clean up zip file
    rm data/mit_indoor.zip
    
    echo "Dataset downloaded and extracted successfully!"
else
    echo "MIT indoor directory already exists."
fi

# List of models to train
models=("lenet" "alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "resnet101" "inceptionv3" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for optimizer in "adam" "sgd" "adamw"; do
        echo "=========================================="
        echo "Training model: $model with optimizer: $optimizer"
        echo "=========================================="
        python train.py --config config/${model}_${optimizer}.json --dataset mit_indoor --input_size 224 --batch_size 32
        
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

