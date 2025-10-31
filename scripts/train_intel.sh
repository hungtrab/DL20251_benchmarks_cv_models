# Check if intel_image directory exists
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
models=("resnet18" "resnet34" "resnet50" "vgg16" "alexnet" "mobilenetv3")

# Loop through each model and train
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Training model: $model"
    echo "=========================================="
    python train.py --config config/intel_resnet18.json --model_name "$model"
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed training for $model"
    else
        echo "✗ Training failed for $model"
    fi
    echo ""
done

echo "All training jobs completed!"

