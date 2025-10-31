"""
Usage examples for the updated prepare_data function.
Demonstrates how to load Intel, MIT Indoor, and ImageNet64 datasets.
"""

from data_preprocess import prepare_data

# ============================================================================
# Example 1: Intel Image Dataset (standard ImageFolder structure)
# ============================================================================
print("=" * 70)
print("Example 1: Intel Image Dataset")
print("=" * 70)

intel_train_dir = 'data/intel_image/seg_train/seg_train'
intel_test_dir = 'data/intel_image/seg_test/seg_test'

dataloaders, dataset_sizes, class_names, num_classes = prepare_data(
    train_dir=intel_train_dir,
    test_dir=intel_test_dir,
    input_size=150,
    batch_size=32,
    dataset='intel'
)

print(f"Classes: {class_names}")
print()

# ============================================================================
# Example 2: MIT Indoor Dataset (single folder + text files)
# ============================================================================
print("=" * 70)
print("Example 2: MIT Indoor Dataset")
print("=" * 70)

# train_dir should point to the Images folder
# test_dir should point to TestImages.txt
mit_images_dir = 'dataset_playground/indoorCVPR_09/Images'
mit_test_txt = 'dataset_playground/TestImages.txt'

dataloaders, dataset_sizes, class_names, num_classes = prepare_data(
    train_dir=mit_images_dir,
    test_dir=mit_test_txt,
    input_size=224,
    batch_size=32,
    dataset='mit_indoor'
)

print(f"Number of classes: {num_classes}")
print(f"First 5 classes: {class_names[:5]}")
print()

# ============================================================================
# Example 3: ImageNet64 Dataset (pickled batch files)
# ============================================================================
# print("=" * 70)
# print("Example 3: ImageNet64 Dataset")
# print("=" * 70)

# # Option A: Pass folder containing batch files
# imagenet_train_dir = 'dataset_playground/Imagenet64_train_part1'
# imagenet_test_dir = 'dataset_playground/Imagenet64_train_part1'  # or separate test folder

# dataloaders, dataset_sizes, class_names, num_classes = prepare_data(
#     train_dir=imagenet_train_dir,
#     test_dir=imagenet_test_dir,
#     input_size=64,  # ImageNet64 is already 64x64
#     batch_size=128,
#     dataset='imagenet64'
# )

# print(f"Number of classes: {num_classes}")
# print()

# Option B: Pass specific batch file paths
train_batches = [
    'dataset_playground/Imagenet64_train_part1/train_data_batch_1',
    # 'dataset_playground/Imagenet64_train_part1/train_data_batch_2',
    # 'dataset_playground/Imagenet64_train_part1/train_data_batch_3',
]
test_batches = [
    'dataset_playground/Imagenet64_train_part1/train_data_batch_4',
]

dataloaders, dataset_sizes, class_names, num_classes = prepare_data(
    train_dir=train_batches,
    test_dir=test_batches,
    input_size=64,
    batch_size=128,
    dataset='imagenet64'
)

# ============================================================================
# Using the dataloaders in training
# ============================================================================
print("=" * 70)
print("Example: Iterating through a dataloader")
print("=" * 70)

# # Get one batch
for images, labels in dataloaders['train']:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    break
