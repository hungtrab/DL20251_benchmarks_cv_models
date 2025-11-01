# Model and Dataset Configuration Guide

This guide provides essential information about datasets, models, and their configurations for the DL20251 Benchmarks CV Models project.

---

## 1. Dataset Image Sizes

### Built-in Torchvision Datasets

| Dataset | Original Image Size | Recommended Input Size | Number of Classes | Notes |
|---------|-------------------|----------------------|------------------|-------|
| **MNIST** | 28×28 (grayscale) | 28×28 | 10 | Handwritten digits |
| **FashionMNIST** | 28×28 (grayscale) | 28×28 | 10 | Fashion items |
| **CIFAR-100** | 32×32 (RGB) | 32×32 | 100 | Natural images, 100 fine-grained classes |
| **Caltech101** | Variable | 224×224 | 101 | Object recognition, resized to 224×224 |

### Custom Datasets

| Dataset | Original Image Size | Recommended Input Size | Number of Classes | Notes |
|---------|-------------------|----------------------|------------------|-------|
| **Intel Image Classification** | Variable | 224×224 | 6 | Natural scenes (buildings, forest, glacier, mountain, sea, street) |
| **MIT Indoor Scenes** | Variable | 224×224 | 67 | Indoor scene categories |
| **ImageNet64** | 64×64 (RGB) | 64×64 or 224×224 | 1000 | Downsampled ImageNet |

---

## 2. Model Hyperparameters

### LeNet
- **Input Channels**: 1 (grayscale) or 3 (RGB)
- **Recommended Input Size**: 28×28 or 32×32
- **Suitable Datasets**: MNIST, FashionMNIST
- **Parameters**:
  - `num_classes`: Number of output classes (default: 10)
  - `in_channels`: 1 for grayscale, 3 for RGB

### AlexNet
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 227×227 (original) or 224×224
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3

### VGG16 / VGG16-BatchNorm
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 224×224
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, CIFAR-100, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3
  - `dropout_rate`: 0.4-0.5 (default: 0.5)
  - `input_size`: 224 (used for adaptive pooling)

### ResNet (18/34/50/101)
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 224×224
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, CIFAR-100, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3
- **Architecture Variations**:
  - ResNet18: 2-2-2-2 blocks (fastest)
  - ResNet34: 3-4-6-3 blocks
  - ResNet50: 3-4-6-3 bottleneck blocks
  - ResNet101: 3-4-23-3 bottleneck blocks (deepest)

### InceptionV1
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 224×224
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3

### InceptionV3
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 299×299 (minimum 75×75)
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3
  - `aux_logits`: True/False (auxiliary classifier for training)

### MobileNetV3
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 224×224
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, CIFAR-100, ImageNet64
- **Parameters**:
  - `mode`: 'large' or 'small'
  - `num_classes`: Number of output classes (default: 1000)
  - `dropout`: 0.2 (default)
- **Modes**:
  - **Large**: Higher accuracy, slower
  - **Small**: Faster, lower accuracy

### Vision Transformer (ViT)
- **Input Channels**: 3 (RGB)
- **Recommended Input Size**: 224×224 (must be divisible by patch_size)
- **Suitable Datasets**: Intel, MIT Indoor, Caltech101, ImageNet64
- **Parameters**:
  - `num_classes`: Number of output classes (default: 1000)
  - `in_channels`: 3
  - `img_size`: 224 (default)
  - `patch_size`: 16 (default)
  - `emb_dim`: 768 (embedding dimension)
  - `num_layers`: 12 (transformer layers)
  - `num_heads`: 12 (attention heads)
  - `ratio_dim`: 4 (MLP expansion ratio)
  - `dropout_rate`: 0.1

---

## 3. Dataset-Model Compatibility

### MNIST / FashionMNIST (28×28, grayscale)
**Recommended Models:**
- ✅ **LeNet** (designed for 28×28 grayscale)
- ⚠️ AlexNet (requires adaptation for grayscale)
- ⚠️ VGG16 (overkill for simple datasets)

**Configuration:**
```json
{
  "dataset_info": {
    "dataset": "mnist",
    "batch_size": 64
  },
  "model_info": {
    "name": "lenet"
  },
  "train_info": {
    "num_epochs": 5,
    "learning_rate": 0.001,
    "optimizer": "adam"
  }
}
```

### CIFAR-100 (32×32, RGB)
**Recommended Models:**
- ✅ **ResNet18/34** (good balance)
- ✅ **MobileNetV3-Small** (efficient)
- ✅ **VGG16** (with small input size)
- ⚠️ InceptionV3 (input size too small, minimum 75×75)

**Configuration:**
```json
{
  "dataset_info": {
    "dataset": "cifar100",
    "input_size": 32,
    "batch_size": 128
  },
  "model_info": {
    "name": "resnet18"
  },
  "train_info": {
    "num_epochs": 100,
    "learning_rate": 0.1,
    "optimizer": "sgd",
    "scheduler": "cosine"
  }
}
```

### Intel / MIT Indoor / Caltech101 (224×224, RGB)
**Recommended Models:**
- ✅ **ResNet18/34/50** (best overall)
- ✅ **MobileNetV3** (efficient, mobile-friendly)
- ✅ **VGG16-BN** (good baseline)
- ✅ **InceptionV3** (high accuracy, requires 299×299)
- ✅ **ViT** (best for large datasets)

**Configuration:**
```json
{
  "dataset_info": {
    "dataset": "intel",
    "input_size": 224,
    "batch_size": 32
  },
  "model_info": {
    "name": "resnet18",
    "dropout_rate": 0.4
  },
  "train_info": {
    "num_epochs": 25,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "cosine"
  }
}
```

### ImageNet64 (64×64 or 224×224, RGB, 1000 classes)
**Recommended Models:**
- ✅ **ResNet50/101** (strong performance)
- ✅ **InceptionV3** (multi-scale features)
- ✅ **ViT** (state-of-the-art)
- ✅ **MobileNetV3-Large** (efficient)

---

## 4. Torchvision Dataset Directory Configuration

### Default Behavior
By default, torchvision datasets are downloaded to the path specified in `data_dir` parameter. The datasets are automatically stored in subdirectories.

### Changing Dataset Directory

**Method 1: Modify the data_dir in config file**
```json
{
  "dataset_info": {
    "dataset": "mnist",
    "data_dir": "data/mnist",  // Custom directory
    "batch_size": 64
  }
}
```

**Method 2: Use command-line argument**
```bash
python train.py --dataset mnist --data_dir ./data/mnist
```

### Directory Structure for Torchvision Datasets

After first download, the structure will be:
```
data/
├── mnist/
│   ├── MNIST/
│   │   └── raw/
│   │       ├── train-images-idx3-ubyte
│   │       ├── train-labels-idx1-ubyte
│   │       ├── t10k-images-idx3-ubyte
│   │       └── t10k-labels-idx1-ubyte
│   └── processed/
├── fashionmnist/
│   └── FashionMNIST/
├── cifar100/
│   └── cifar-100-python/
└── caltech101/
    └── caltech101/
```

### Avoiding Re-download

**Important:** Torchvision datasets check if the data already exists before downloading.

**To avoid re-downloading:**

1. **Keep the same directory structure**: Don't rename or move the dataset folders
2. **Use the same `data_dir` path**: Always point to the parent directory containing the dataset

**Example:**

```python
# First time - downloads data
train_dataset = datasets.MNIST(root='data/mnist', train=True, download=True)

# Subsequent runs - uses existing data (download=True is safe, it won't re-download)
train_dataset = datasets.MNIST(root='data/mnist', train=True, download=True)

# You can also set download=False after first run
train_dataset = datasets.MNIST(root='data/mnist', train=True, download=False)
```

**In train.py:**

The `prepare_builtin_data()` function automatically handles this:
```python
# Always set download=True - it only downloads if data doesn't exist
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
```

### Moving Downloaded Data

If you want to move or share downloaded datasets:

1. **Copy the entire dataset directory:**
   ```bash
   # Example: Copy MNIST dataset
   cp -r data/mnist /new/location/mnist
   ```

2. **Update your config or command to point to new location:**
   ```bash
   python train.py --dataset mnist --data_dir /new/location/mnist
   ```

3. **Or create symbolic link:**
   ```bash
   ln -s /existing/data/mnist data/mnist
   ```

### Dataset Storage Requirements

| Dataset | Approximate Size | Download Time (fast internet) |
|---------|-----------------|------------------------------|
| MNIST | ~50 MB | < 1 minute |
| FashionMNIST | ~60 MB | < 1 minute |
| CIFAR-100 | ~170 MB | 1-2 minutes |
| Caltech101 | ~130 MB | 1-2 minutes |

### Best Practices

1. **Centralized Data Directory**: Keep all datasets in one `data/` folder
2. **Version Control**: Add `data/` to `.gitignore` to avoid committing large files
3. **Shared Storage**: On shared systems, use a common data directory:
   ```bash
   export TORCH_DATA_DIR=/shared/datasets
   python train.py --dataset mnist --data_dir $TORCH_DATA_DIR/mnist
   ```
4. **Environment Variable**: Set a default data directory:
   ```bash
   # In your .bashrc or .zshrc
   export PYTORCH_DATA_ROOT="/path/to/datasets"
   ```

---

## 5. Training Configuration Examples

### Quick Test (Fast Training)
```bash
python train.py --dataset mnist --model_name lenet --num_epochs 2 --batch_size 128
```

### Full Training (Intel Dataset with ResNet18)
```bash
python train.py --config config/intel_resnet18.json
```

### Override Config Values
```bash
python train.py --config config/intel_resnet18.json --model_name resnet50 --num_epochs 50
```

### Training Multiple Models (using script)
```bash
bash scripts/train_intel.sh
```

---

## 6. Common Issues and Solutions

### Issue: "Dataset not found"
**Solution:** Make sure `download=True` is set for torchvision datasets, or check the path for custom datasets.

### Issue: CUDA out of memory
**Solutions:**
- Reduce batch size: `--batch_size 16`
- Use a smaller model: Switch from ResNet50 to ResNet18
- Reduce input size: `--input_size 128`

### Issue: Re-downloading datasets
**Solutions:**
- Ensure you're using the same `data_dir` path
- Check that the dataset files haven't been deleted or moved
- Verify directory permissions

### Issue: Model too large for small datasets
**Solution:** Use appropriate models:
- MNIST/FashionMNIST → LeNet
- CIFAR-100 → ResNet18, MobileNetV3-Small
- Large datasets → ResNet50, InceptionV3, ViT

---

## 7. Recommended Training Strategies

### For Small Datasets (< 10K images)
- Use smaller models (LeNet, ResNet18)
- Higher learning rate: 0.001-0.01
- Fewer epochs: 10-25
- Data augmentation: Essential to prevent overfitting

### For Medium Datasets (10K-100K images)
- Medium models (ResNet34/50, VGG16, MobileNetV3)
- Moderate learning rate: 0.0001-0.001
- More epochs: 25-100
- Learning rate scheduling: Cosine annealing

### For Large Datasets (> 100K images)
- Large models (ResNet101, InceptionV3, ViT)
- Lower learning rate: 0.00001-0.0001
- Extended training: 100-300 epochs
- Advanced techniques: Mixed precision, distributed training

---

## Quick Reference Table

| Model | Best For | Input Size | Speed | Accuracy | Parameters |
|-------|----------|------------|-------|----------|------------|
| LeNet | MNIST, simple tasks | 28×28 | ⚡⚡⚡⚡⚡ | ⭐⭐ | ~60K |
| AlexNet | Baseline CNN | 224×224 | ⚡⚡⚡⚡ | ⭐⭐⭐ | ~60M |
| VGG16 | Feature extraction | 224×224 | ⚡⚡ | ⭐⭐⭐⭐ | ~138M |
| ResNet18 | General purpose | 224×224 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ~11M |
| ResNet50 | High accuracy | 224×224 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ~25M |
| InceptionV3 | Multi-scale | 299×299 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ~24M |
| MobileNetV3 | Mobile/Edge | 224×224 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ~5M |
| ViT | Large datasets | 224×224 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ~86M |

---

**Last Updated:** November 1, 2025
