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
models=("alexnet" "resnet18" "resnet34" "resnet50" "vgg16" "mobilenetv3_l" "mobilenetv3_s" "vit" "efficientnetv2_s" "efficientnetv2_m" "efficientnetv2_l" "vgg16_bn")

# Loop through each model and train
for model in "${models[@]}"; do
    for scheduler in "constant" "cosine"; do
        echo "=========================================="
        echo "Training model: $model with scheduler: $scheduler"
        echo "=========================================="
        python train.py --config config/${model}_adamw.json --dataset mit --input_size 224 --batch_size 64 --num_warmup_steps 500 --use_tensorboard --tensorboard_log_dir results/tensorboard/${model}_${scheduler}_mit --use_wandb --wandb_project dl20251-cv --wandb_run_name ${model}_${scheduler}_mit --scheduler $scheduler --num_epochs 40
        
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

