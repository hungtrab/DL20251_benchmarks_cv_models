import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import pickle
from PIL import Image

# ===================== Custom Dataset Classes =====================

class MITIndoorDataset(Dataset):
    """
    Dataset for MIT Indoor scenes with train/test split from text files.
    Images are in a single folder, split defined by TrainImages.txt and TestImages.txt
    """
    def __init__(self, image_dir, split_file, transform=None):
        """
        Args:
            image_dir: Root directory containing Images folder with all category subfolders
            split_file: Path to TrainImages.txt or TestImages.txt
            transform: Optional transform to be applied on images
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Read image paths from split file
        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Extract class names from image paths (format: classname/image.jpg)
        self.class_names = sorted(list(set([path.split('/')[0] for path in self.image_paths])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Get label from path
        class_name = self.image_paths[idx].split('/')[0]
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Imagenet64Dataset(Dataset):
    """
    Dataset for ImageNet64 pickled batch files.
    Handles multiple batch files with lazy loading to save memory.
    """
    def __init__(self, batch_paths, transform=None, preload=False):
        """
        Args:
            batch_paths: list of paths to pickled batch files or single path
            transform: torchvision transform to apply
            preload: if True, load all data into memory; if False, lazy load
        """
        if isinstance(batch_paths, str):
            batch_paths = [batch_paths]
        self.batch_paths = batch_paths
        self.transform = transform
        self.preload = bool(preload)
        
        # Build metadata
        self._batch_meta = []  # list of (path, n_samples)
        self._total_len = 0
        
        if self.preload:
            # Load everything into memory
            data_list = []
            labels_list = []
            for p in batch_paths:
                with open(p, 'rb') as f:
                    b = pickle.load(f)
                data_list.append(b['data'])
                labels_list.extend(b['labels'])
            self._data = np.vstack(data_list)
            self._labels = np.array(labels_list, dtype=np.int64)
            self._total_len = self._data.shape[0]
        else:
            # Lazy mode: just get counts
            for p in batch_paths:
                with open(p, 'rb') as f:
                    b = pickle.load(f)
                n = b['data'].shape[0]
                self._batch_meta.append((p, n))
                self._total_len += n
            
            # Cache for current batch
            self._current_batch_idx = None
            self._current_data = None
            self._current_labels = None
    
    def __len__(self):
        return self._total_len
    
    def _load_batch(self, batch_idx):
        """Load a batch file into cache (lazy mode only)."""
        if self._current_batch_idx == batch_idx:
            return
        
        path, n = self._batch_meta[batch_idx]
        with open(path, 'rb') as f:
            b = pickle.load(f)
        self._current_data = b['data']
        self._current_labels = b['labels']
        self._current_batch_idx = batch_idx
    
    def __getitem__(self, idx):
        # Get row and label
        if self.preload:
            row = self._data[idx]
            label = int(self._labels[idx])
        else:
            # Find which batch contains this index
            offset = 0
            for i, (p, n) in enumerate(self._batch_meta):
                if idx < offset + n:
                    inner_idx = idx - offset
                    self._load_batch(i)
                    row = self._current_data[inner_idx]
                    label = int(self._current_labels[inner_idx])
                    break
                offset += n
            else:
                raise IndexError(idx)
        
        # Convert row to image: (12288,) -> (3, 64, 64) -> (64, 64, 3)
        img = row.reshape(3, 64, 64)
        img = np.transpose(img, (1, 2, 0))  # HWC for PIL
        
        # Ensure uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        
        if self.transform:
            return self.transform(pil_img), torch.tensor(label, dtype=torch.long)
        else:
            # Default: convert to tensor
            tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
            return tensor, torch.tensor(label, dtype=torch.long)
    
    def reshuffle(self, seed=None):
        """
        Re-shuffle the global index map. Call this at the start of each epoch
        to get a different shuffle order.
        
        Args:
            seed: optional random seed. If None, uses current RNG state.
        """
        rng = np.random.RandomState(seed)
        rng.shuffle(self._index_map)
        print(f'Index map reshuffled (seed={seed})')


# ===================== Data Preparation Functions =====================

def prepare_data(train_dir, test_dir, input_size, batch_size, dataset='intel'):
    """
    Prepare data loaders for different dataset formats.
    
    Args:
        train_dir: Directory/path for training data
                   - intel: path to train folder with class subfolders
                   - mit_indoor: path to Images folder
                   - imagenet64: path to folder containing batch files OR list of batch file paths
        test_dir: Directory/path for testing data
                  - intel: path to test folder with class subfolders
                  - mit_indoor: path to TestImages.txt
                  - imagenet64: path to folder containing batch files OR list of batch file paths
        input_size: Input image size (images will be resized to this)
        batch_size: Batch size for dataloaders
        dataset: Dataset type - 'intel', 'mit_indoor', or 'imagenet64'
        
    Returns:
        tuple: (dataloaders, dataset_sizes, class_names, num_classes)
    """
    dataset = dataset.lower()
    
    # ============ Intel Image Dataset (standard ImageFolder structure) ============
    if dataset == 'intel' or dataset == 'intel_image':
        print("Loading Intel Image Dataset...")
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        image_datasets = {
            'train': datasets.ImageFolder(root=train_dir, transform=data_transforms['train']),
            'test': datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])
        }
        
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
    
    # ============ MIT Indoor Dataset (single folder + text files) ============
    elif dataset == 'mit_indoor' or dataset == 'mit':
        print("Loading MIT Indoor Dataset...")
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        # train_dir should be path to Images folder
        # test_dir should be path to TestImages.txt
        # Infer TrainImages.txt path
        if os.path.isfile(test_dir):
            # test_dir is the TestImages.txt file
            test_txt = test_dir
            train_txt = os.path.join(os.path.dirname(test_dir), 'TrainImages.txt')
            images_dir = train_dir  # train_dir is the Images folder
        else:
            raise ValueError("For MIT Indoor dataset, test_dir should be path to TestImages.txt")
        
        image_datasets = {
            'train': MITIndoorDataset(images_dir, train_txt, transform=data_transforms['train']),
            'test': MITIndoorDataset(images_dir, test_txt, transform=data_transforms['test'])
        }
        
        class_names = image_datasets['train'].class_names
        num_classes = len(class_names)
    
    # ============ ImageNet64 Dataset (pickled batch files) ============
    elif dataset == 'imagenet64' or dataset == 'imagenet':
        print("Loading ImageNet64 Dataset...")
        
        # Compute mean from first batch if available
        if isinstance(train_dir, list):
            first_batch = train_dir[0]
        elif os.path.isdir(train_dir):
            # train_dir is a folder, find batch files
            batch_files = sorted([f for f in os.listdir(train_dir) if 'batch' in f.lower()])
            train_dir = [os.path.join(train_dir, f) for f in batch_files]
            first_batch = train_dir[0]
        else:
            first_batch = train_dir
            train_dir = [train_dir]
        
        # Load mean from first batch
        try:
            with open(first_batch, 'rb') as f:
                b = pickle.load(f)
            flat_mean = b.get('mean')
            if flat_mean is not None:
                flat_mean = np.asarray(flat_mean)
                per_chan_mean = flat_mean.reshape(3, 64, 64).mean(axis=(1, 2)) / 255.0
                per_chan_mean = per_chan_mean.tolist()
            else:
                per_chan_mean = [0.485, 0.456, 0.406]
        except Exception as e:
            print(f"Could not load mean from batch file: {e}")
            per_chan_mean = [0.485, 0.456, 0.406]
        
        # ImageNet64 is already 64x64, no need to resize
        # Add augmentation for training
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=per_chan_mean, std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=per_chan_mean, std=[0.229, 0.224, 0.225])
            ])
        }
        
        # Handle test_dir similarly
        if isinstance(test_dir, list):
            test_paths = test_dir
        elif os.path.isdir(test_dir):
            batch_files = sorted([f for f in os.listdir(test_dir) if 'batch' in f.lower()])
            test_paths = [os.path.join(test_dir, f) for f in batch_files]
        else:
            test_paths = [test_dir]
        
        image_datasets = {
            'train': Imagenet64Dataset(train_dir, transform=data_transforms['train'], preload=False),
            'test': Imagenet64Dataset(test_paths, transform=data_transforms['test'], preload=False)
        }
        
        # ImageNet64 has 1000 classes (standard ImageNet)
        class_names = [str(i) for i in range(1000)]
        num_classes = 1000
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available options: 'intel', 'mit_indoor', 'imagenet64'")
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': len(image_datasets['train']),
        'test': len(image_datasets['test'])
    }
    
    print(f"{dataset.upper()} Data Prepared:")
    print("Number of classes:", num_classes)
    print("Train set size:", dataset_sizes['train'])
    print("Test set size:", dataset_sizes['test'])
    
    return dataloaders, dataset_sizes, class_names, num_classes

def prepare_builtin_data(data_dir, batch_size, dataset='mnist'):
    """
    Prepare data loaders for built-in datasets.
    
    Args:
        data_dir: Directory to download/load dataset
        batch_size: Batch size for dataloaders
        dataset: Name of the dataset to prepare. Options: 'mnist', 'fashionmnist', 'cifar100', 'caltech101'

    Returns:
        tuple: (dataloaders, dataset_sizes, class_names, num_classes)
    """
    dataset = dataset.lower()
    
    # Define transforms based on dataset
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), 
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
        num_classes = 10
        
    elif dataset == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # FashionMNIST mean and std
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        num_classes = 10
        
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
        # CIFAR-100 has 100 fine-grained classes
        class_names = train_dataset.classes
        num_classes = 100
        
    elif dataset == 'caltech101':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        full_dataset = datasets.Caltech101(root=data_dir, download=True, transform=transform)
        
        # Split Caltech101 into train and test (80/20 split)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        class_names = full_dataset.categories
        num_classes = len(class_names)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available options: 'mnist', 'fashionmnist', 'cifar100', 'caltech101'")
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }
    
    print(f"{dataset.upper()} Data Prepared:")
    print("Number of classes:", num_classes)
    print("Train set size:", dataset_sizes['train'])
    print("Test set size:", dataset_sizes['test'])
    
    return dataloaders, dataset_sizes, class_names, num_classes