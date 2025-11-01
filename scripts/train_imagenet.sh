#!/bin/bash

# Check if imagenet directory exists
cd ..

if [ ! -d "data/imagenet" ]; then
    echo "ImageNet dataset not found. Downloading..."

    # Create data directory if it doesn't exist
    mkdir -p data
    
    # Install gdown if not already installed
    pip install -q gdown
    
    # Download the dataset

    gdown https://drive.google.com/uc?id=17F3AhEYP2t9YYXV_3bvcG6laEGScajOV -O data/imagenet_train_p1.zip
    gdown https://drive.google.com/uc?id=1kpJXAE5T3TmG-vW-orkbh1ssYHtop5OY -O data/imagenet_train_p2.zip
    gdown https://drive.google.com/uc?id=1Abyp4o6TYG4RsoXZ3jrnbwK_E3hXOAQY -O data/imagenet_test.zip
    # Unzip the dataset
    echo "Extracting dataset..."
    unzip data/imagenet_train_p1.zip -d data/imagenet
    unzip data/imagenet_train_p2.zip -d data/imagenet
    unzip data/imagenet_test.zip -d data/imagenet

    # Clean up zip file
    rm data/imagenet_train_p1.zip
    rm data/imagenet_train_p2.zip
    rm data/imagenet_test.zip

    echo "Dataset downloaded and extracted successfully!"
else
    echo "Imagenet directory already exists."
fi

# List of models to train
models=("lenet" "alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "resnet101" "inceptionv3" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for optimizer in "adam" "sgd" "adamw"; do
        echo "=========================================="
        echo "Training model: $model with optimizer: $optimizer"
        echo "=========================================="
        python train.py --config config/${model}_${optimizer}.json --dataset imagenet --input_size 64 --batch_size 32
        
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

