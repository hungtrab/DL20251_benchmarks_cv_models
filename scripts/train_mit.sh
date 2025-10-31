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

