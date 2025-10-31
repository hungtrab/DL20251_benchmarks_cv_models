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
    echo "Intel directory already exists. Running training script..."
    # Run the training script
fi

python train.py \
    --dataset='intel' \
    --input_size=224 \
    --batch_size=64 \
    --num_epochs=2 \
    --learning_rate=1e-4 \
    --model_name="resnet18" \
    --dropout_rate=0.4 