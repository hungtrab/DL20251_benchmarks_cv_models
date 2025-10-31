import os
import time
from evaluate import evaluate_model
from trainer import Trainer, count_images_per_class, calculate_class_weights
from data_preprocess import prepare_data, prepare_builtin_data
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import json
from pathlib import Path
from typing import Any, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from evaluate import evaluate_model

def _flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested config into argparse-compatible keys.

    Supported sections: dataset.*, model.*, train.* map to corresponding CLI args.
    Top-level keys that match CLI names are also accepted.
    """
    flat: Dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return flat
    
    # Map nested config sections to argparse argument names
    section_map = {
        'dataset_info': {
            'dataset': 'dataset',
            'input_size': 'input_size',
            'batch_size': 'batch_size',
        },
        'model_info': {
            'name': 'model_name',
            'type': 'model_type',
            'dropout_rate': 'dropout_rate',
            'pretrained': 'pretrained',
        },
        'train_info': {
            'num_epochs': 'num_epochs',
            'learning_rate': 'learning_rate',
            'optimizer': 'optimizer',
            'criterion': 'criterion',
            'scheduler': 'scheduler',
            'num_warmup_steps': 'num_warmup_steps',
            'use_class_weights': 'use_class_weights',
            'weight_type': 'weight_type',
            'seed': 'seed',
        },
    }
    
    # Process nested sections
    for section, mapping in section_map.items():
        if section in cfg and isinstance(cfg[section], dict):
            for k, arg_name in mapping.items():
                if k in cfg[section]:
                    flat[arg_name] = cfg[section][k]
    
    # Also accept top-level keys that match argparse names (for backward compatibility)
    for key in [
        'dataset', 'input_size', 'batch_size',
        'model_name', 'model_type', 'dropout_rate', 'pretrained',
        'num_epochs', 'learning_rate', 'optimizer', 'criterion', 'scheduler',
        'num_warmup_steps', 'use_class_weights', 'weight_type', 'seed'
    ]:
        if key in cfg:
            flat[key] = cfg[key]
    
    return flat


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Example training script")
    parser.add_argument('--config', type=str, default=None, help='Path to a JSON config file')
    # parser.add_argument('--train_dir', type=str, help='Path to the training data directory')
    # parser.add_argument('--test_dir', type=str, help='Path to the testing data directory')
    # parser.add_argument('--mnist_data_dir', type=str, default=None, help='Directory to store MNIST data')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'intel', 'fashionmnist', 'cifar100', 'mit', 'imagenet', 'caltech101'],
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type=str, default='alexnet',
                        choices=['linearsvm_mnist', 'alexnet', 'vgg16', 'lenet', 'vgg16', 'vgg16_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'inceptionv3', 'mobilenetv3', 'vit'],
                        help='Name of the model to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')
    # parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--criterion', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'hinge'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='constant', choices=['constant', 'linear', 'cosine'], help='Learning rate scheduler to use')
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='Number of warmup steps for the scheduler')
    parser.add_argument('--model_type', type=str, default='large', choices=['large', 'small'], help='Model type of MobileNetV3')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate for model')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss function')
    parser.add_argument('--weight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], help='Type of class weights to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # First, parse arguments to identify which were explicitly provided by the user
    # We'll use parse_known_args to get the namespace and also track what was provided
    import sys
    
    # Determine which arguments were explicitly provided on command line
    # by checking if they appear in the input arguments
    if input_args is None:
        input_args = sys.argv[1:]
    
    # Track which arguments were explicitly provided
    provided_args = set()
    i = 0
    while i < len(input_args):
        arg = input_args[i]
        if arg.startswith('--'):
            arg_name = arg[2:]  # Remove '--' prefix
            # Handle both --arg=value and --arg value formats
            if '=' in arg_name:
                arg_name = arg_name.split('=')[0]
            provided_args.add(arg_name)
        i += 1
    
    # Parse all arguments
    args = parser.parse_args(input_args)

    # Merge config values, letting explicit CLI flags override config
    if args.config is not None:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            try:
                raw_cfg = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON config: {e}")
        flat_cfg = _flatten_config(raw_cfg)
        print(f"Loaded config from {cfg_path}: {flat_cfg}")
        
        # For each config key, only apply if the argument was NOT explicitly provided by user
        for dest, value in flat_cfg.items():
            if dest not in vars(args):
                continue
            # Only set from config if user didn't explicitly provide this argument
            if dest not in provided_args:
                setattr(args, dest, value)
            else:
                print(f"  CLI override: --{dest} = {getattr(args, dest)} (config has: {value})")

    return args

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # if args.mnist_data_dir is not None:
    #     dataloaders, dataset_sizes, class_names, num_classes = prepare_mnist_data(data_dir=args.mnist_data_dir, batch_size=args.batch_size)
    # else:
    #     if not args.train_dir or not args.test_dir:
    #         raise ValueError("train_dir and test_dir must be specified for models other than linearsvm_mnist")
    #     dataloaders, dataset_sizes, class_names, num_classes = prepare_data(train_dir= args.train_dir, test_dir= args.test_dir, input_size= args.input_size, batch_size= args.batch_size)
    result_path = os.path.abspath('results')
    os.makedirs(result_path, exist_ok=True)
    exp_name = f"{args.dataset}_{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.mkdir(os.path.join(result_path, exp_name))
    if args.dataset in ['mnist', 'fashionmnist', 'cifar100', 'caltech101']:
        dataloaders, dataset_sizes, class_names, num_classes = prepare_builtin_data(data_dir=args.dataset, batch_size=args.batch_size, dataset=args.dataset)
    elif args.dataset in ['intel', 'mit', 'imagenet']:
        if args.dataset == 'intel':
            train_dir = 'data/intel_image/seg_train/seg_train'
            test_dir = 'data/intel_image/seg_test/seg_test'
        elif args.dataset == 'mit':
            train_dir = 'data/mit_indoor/indoorCVPR_09/Images'
            test_dir = 'data/mit_indoor/TestImages.txt'
        elif args.dataset == 'imagenet':
            train_dir = [
                'dataset_playground/Imagenet64_train_part1/train_data_batch_1',
                # 'dataset_playground/Imagenet64_train_part1/train_data_batch_2',
                # 'dataset_playground/Imagenet64_train_part1/train_data_batch_3',
            ]
            test_dir = [
                'dataset_playground/Imagenet64_train_part1/train_data_batch_4',
            ]
        dataloaders, dataset_sizes, class_names, num_classes = prepare_data(train_dir= train_dir, test_dir= test_dir, input_size= args.input_size, batch_size= args.batch_size, dataset=args.dataset)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")
    
    if args.model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    elif args.model_name == 'lenet':
        model = LeNet(num_classes=num_classes, in_channels=1)
    elif args.model_name == 'vgg16':
        model = VGG16(num_classes = num_classes, in_channels = 3, dropout_rate= 0.4, input_size=args.input_size)
    elif args.model_name == 'vgg16_bn':
        model = VGG16BatchNorm(num_classes= num_classes, in_channels = 3, dropout_rate= 0.4, input_size=args.input_size)
    elif args.model_name == 'resnet18':
        model = resnet18(num_classes = num_classes, in_channels= 3)
    elif args.model_name == 'resnet34':
        model = resnet34(num_classes = num_classes, in_channels= 3)
    elif args.model_name == 'resnet50':
        model = resnet50(num_classes= num_classes, in_channels= 3)
    elif args.model_name == 'resnet101':
        model = resnet101(num_classes= num_classes, in_channels= 3)
    elif args.model_name == 'inceptionv3':
        model = InceptionV3(num_classes=num_classes, in_channels=3)
    elif args.model_name == 'mobilenetv3':
        model = MobileNetV3(mode = args.model_type, num_classes = num_classes, dropout=args.dropout_rate)
    elif args.model_name == 'vit':
        model = VisionTransformer(num_classes = num_classes, dropout_rate= args.dropout_rate)
    else:
        raise ValueError(f"Model {args.model_name} not recognized.")
    print(f"Model: {model}")
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")
    
    if args.criterion == 'cross_entropy':
        if args.use_class_weights:
            class_counts = count_images_per_class(dataloaders['train'])
            class_weights = calculate_class_weights(class_counts, weight_type=args.weight_type)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion == "hinge":
        criterion = nn.MultiMarginLoss()      
    elif args.criterion == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {args.criterion} not recognized.")
        
    if args.scheduler == 'constant':
        scheduler = None
    else:
        warmup_steps = args.num_warmup_steps
        total_steps = args.num_epochs * len(dataloaders['train'])
        decay_steps = int(0.05 * total_steps) 
        steady_steps = total_steps - warmup_steps - decay_steps

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        steady_scheduler = ConstantLR(
            optimizer, factor=1.0, total_iters=steady_steps
        )

        if args.scheduler == 'linear':
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.05,
                total_iters=decay_steps
            )
        elif args.scheduler == 'cosine':
            eta_min = 0.1 * args.learning_rate
            decay_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,
                eta_min=eta_min
            )
        else:
            raise ValueError(f"Unknown scheduler type: {args.scheduler}")

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, steady_scheduler, decay_scheduler],
            milestones=[warmup_steps, warmup_steps + steady_steps]
        )
    best_model_path = os.path.join(result_path, exp_name, 'best_model.pth')
    trainer = Trainer(model, dataloaders= dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=args.num_epochs, save_path=best_model_path)
    model, history = trainer.train()
    # trainer.plot_history()
    trainer.save_history(os.path.join(result_path, exp_name, 'training_history.json'))
    trainer.save_plot_image(os.path.join(result_path, exp_name, 'training_history.png'))
    evaluate_model(model, dataloaders['test'], num_class = num_classes, save_path=os.path.join(result_path, exp_name))
if __name__ == "__main__":
    args = parse_args()
    main(args)