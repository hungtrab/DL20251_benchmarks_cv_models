#!/bin/bash

# Check if intel_image directory exists
cd ..

if [ ! -d "data/intel_image" ]; then
    echo "Intel dataset not found. Downloading..."
    
    # Create data directory if it doesn't exist
    mkdir -p data
    
    # Install gdown if not already installed
    pip install -q gdown
    
    # Download the dataset
    gdown https://drive.google.com/uc?id=1asbLz9GcivwJmfRhBq7eI60LjJAayqMG -O data/intel_image.zip
    
    # Unzip the dataset
    echo "Extracting dataset..."
    unzip -q data/intel_image.zip -d data/intel_image
    
    # Clean up zip file
    rm data/intel_image.zip
    
    echo "Dataset downloaded and extracted successfully!"
else
    echo "Intel directory already exists."
fi

# List of models to train
models=("mobilenetv3_l" "resnet18" "resnet34" "efficientnetv2_s" "efficientnetv2_m" "mobilenetv3_s")

# Loop through each model and train
for model in "${models[@]}"; do
    for scheduler in "constant" "cosine"; do
        echo "=========================================="
        echo "Training model: $model with scheduler: $scheduler"
        echo "=========================================="
        python train.py --config config/${model}_adamw.json --dataset intel --input_size 224 --batch_size 32 --use_tensorboard --tensorboard_log_dir results/tensorboard/${model}_${scheduler}_intel --use_wandb --wandb_project dl20251-cv --wandb_run_name ${model}_${scheduler}_intel --scheduler $scheduler --num_epochs 25
        
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

