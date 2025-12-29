import os
import time
from evaluate import evaluate_model
from trainer import Trainer, count_images_per_class, calculate_class_weights
from data_preprocess import prepare_data, prepare_builtin_data
from model import *
from models_dense import convnextv2_tiny, convnextv2_base, FCMAE_Dense
from benchmark_efficiency import benchmark_efficiency
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import json
from pathlib import Path
from typing import Any, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR, StepLR, OneCycleLR
from evaluate import evaluate_model
from torch.utils.tensorboard import SummaryWriter

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
            'step_size': 'step_size',
            'step_gamma': 'step_gamma',
            'max_lr_factor': 'max_lr_factor',
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
        'num_warmup_steps', 'use_class_weights', 'weight_type', 'seed',
        'step_size', 'step_gamma', 'max_lr_factor'
    ]:
        if key in cfg:
            flat[key] = cfg[key]
    
    return flat


def pretrain_convnext_fcmae(encoder, dataloaders, args, device, exp_dir):
    """
    Self-supervised pretraining using FCMAE for ConvNeXt V2.
    
    Args:
        encoder: ConvNeXt V2 encoder model (without classification head)
        dataloaders: Dictionary with 'train' dataloader
        args: Arguments containing pretraining hyperparameters
        device: Device to train on
        exp_dir: Experiment directory for saving checkpoints
    
    Returns:
        Pretrained encoder state_dict
    """
    print(f"\n{'='*60}")
    print(f"FCMAE SELF-SUPERVISED PRETRAINING")
    print(f"Epochs: {args.conv_pretrain_epochs}")
    print(f"Mask Ratio: {args.conv_mask_ratio}")
    print(f"Learning Rate: {args.conv_pretrain_lr}")
    print(f"Weight Decay: {args.conv_pretrain_wd}")
    print(f"{'='*60}\n")
    
    # Create FCMAE model
    fcmae_model = FCMAE_Dense(
        encoder=encoder,
        mask_ratio=args.conv_mask_ratio,
        decoder_embed_dim=512,
        decoder_depth=1,
        patch_size=32
    )
    fcmae_model.to(device)
    
    # Optimizer for pretraining
    pretrain_optimizer = optim.AdamW(
        fcmae_model.parameters(),
        lr=args.conv_pretrain_lr,
        betas=(0.9, 0.95),
        weight_decay=args.conv_pretrain_wd
    )
    
    # Scheduler for pretraining
    warmup_epochs = min(5, args.conv_pretrain_epochs // 10)
    pretrain_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        pretrain_optimizer,
        T_max=args.conv_pretrain_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    train_loader = dataloaders['train']
    
    # Pretraining loop
    for epoch in range(args.conv_pretrain_epochs):
        # Warmup
        if epoch < warmup_epochs:
            lr_scale = min(1., float(epoch + 1) / warmup_epochs)
            for pg in pretrain_optimizer.param_groups:
                pg['lr'] = args.conv_pretrain_lr * lr_scale
        
        fcmae_model.train()
        running_loss = 0.0
        current_lr = pretrain_optimizer.param_groups[0]['lr']
        
        print(f"Pretrain Epoch [{epoch+1}/{args.conv_pretrain_epochs}] (LR: {current_lr:.6f})")
        
        for i, data in enumerate(train_loader):
            if isinstance(data, (tuple, list)):
                images = data[0]
            else:
                images = data
            
            images = images.to(device, non_blocking=True)
            
            pretrain_optimizer.zero_grad()
            loss, _, _ = fcmae_model(images, mask_ratio=args.conv_mask_ratio)
            loss.backward()
            pretrain_optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                avg_loss = running_loss / 50
                print(f"  Batch [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        if epoch >= warmup_epochs:
            pretrain_scheduler.step()
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"  Epoch {epoch+1} Average Loss: {epoch_loss:.4f}\n")
    
    # Save pretrained encoder
    pretrain_save_path = os.path.join(exp_dir, 'fcmae_pretrained_encoder.pth')
    
    # Extract encoder state dict
    if isinstance(fcmae_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state = fcmae_model.module.state_dict()
    else:
        model_state = fcmae_model.state_dict()
    
    encoder_state = {}
    for k, v in model_state.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.', '')
            encoder_state[new_key] = v
    
    torch.save({
        'epoch': args.conv_pretrain_epochs,
        'model_state_dict': encoder_state,
        'config': {
            'mask_ratio': args.conv_mask_ratio,
            'pretrain_lr': args.conv_pretrain_lr,
            'pretrain_wd': args.conv_pretrain_wd,
            'pretrain_epochs': args.conv_pretrain_epochs
        }
    }, pretrain_save_path)
    
    print(f"\n{'='*60}")
    print(f"PRETRAINING COMPLETE")
    print(f"Encoder weights saved to: {pretrain_save_path}")
    print(f"{'='*60}\n")
    
    return encoder_state

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Example training script")
    parser.add_argument('--config', type=str, default=None, help='Path to a JSON config file')
    # parser.add_argument('--train_dir', type=str, help='Path to the training data directory')
    # parser.add_argument('--test_dir', type=str, help='Path to the testing data directory')
    # parser.add_argument('--mnist_data_dir', type=str, default=None, help='Directory to store MNIST data')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'intel', 'fashionmnist', 'cifar100', 'mit', 'imagenet', 'caltech101', 'cifar100_224'],
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model_name', type=str, default='alexnet',
                        choices=['efficientnetv2_s','efficientnetv2_m', 'efficientnetv2_l', 'alexnet', 'vgg16', 'lenet', 'vgg16_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'inceptionv3', 'mobilenetv3_s', 'mobilenetv3_l', 'vit', 'convnextv2_t', 'convnextv2_b'],
                        help='Name of the model to use')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model weights')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained checkpoint for ConvNeXtV2')
    # parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--criterion', type=str, default='cross_entropy', choices=['cross_entropy', 'mse', 'hinge'], help='Loss function to use')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='constant', choices=['constant', 'linear', 'cosine', 'step', 'onecycle'], help='Learning rate scheduler to use')
    parser.add_argument('--num_warmup_steps', type=int, default=500, help='Number of warmup steps for the scheduler')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for step scheduler (in epochs)')
    parser.add_argument('--step_gamma', type=float, default=0.1, help='Gamma for step scheduler')
    parser.add_argument('--max_lr_factor', type=float, default=10.0, help='Max LR multiplier for onecycle scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate for model')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for loss function')
    parser.add_argument('--weight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], help='Type of class weights to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--fasttrain', action='store_true', help='Enable fast training with mixed precision (torch.cuda.amp)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    # Validation set options
    parser.add_argument('--val', action='store_true', help='Enable validation set (split from training data)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size as fraction of training data (default: 0.2)')
    # W&B logging options
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='dl20251-cv', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team or username)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (defaults to experiment name)')
    # TensorBoard logging options
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_log_dir', type=str, default='logs', help='TensorBoard log directory')
    
    # Section 4.3: Anti-Overfitting & Adaptive Learning
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup data augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix data augmentation')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha parameter')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='Probability of applying CutMix')
    parser.add_argument('--use_sam', action='store_true', help='Enable SAM (Sharpness-Aware Minimization) optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05, help='SAM rho parameter for perturbation')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--adaptive_training', action='store_true', help='Enable adaptive training (auto-adjust WD, aug, SAM)')
    parser.add_argument('--adaptive_check_interval', type=int, default=5, help='Check interval for adaptive training (epochs)')
    
    # Evaluation options (Section 6 - Comprehensive Evaluation)
    parser.add_argument('--eval_robustness', action='store_true', help='Evaluate robustness with noise injection (Section 6.1)')
    parser.add_argument('--eval_calibration', action='store_true', help='Evaluate calibration with ECE (Section 6.2)')
    parser.add_argument('--eval_bootstrap', action='store_true', help='Compute bootstrap confidence interval (Section 6.3)')
    parser.add_argument('--eval_full', action='store_true', help='Run full benchmark evaluation (all Section 6 metrics)')
    parser.add_argument('--eval_efficiency', action='store_true', help='Benchmark efficiency (throughput, latency, VRAM, model size)')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='Number of bootstrap samples for CI')
    parser.add_argument('--n_calibration_bins', type=int, default=15, help='Number of bins for ECE calculation')
    
    # ConvNeXt V2 Self-Supervised Pretraining
    parser.add_argument('--conv_pretrain', action='store_true', help='Enable FCMAE self-supervised pretraining for ConvNeXt V2 models')
    parser.add_argument('--conv_pretrain_epochs', type=int, default=50, help='Number of epochs for ConvNeXt V2 pretraining')
    parser.add_argument('--conv_mask_ratio', type=float, default=0.6, help='Masking ratio for FCMAE pretraining')
    parser.add_argument('--conv_pretrain_lr', type=float, default=1.5e-4, help='Learning rate for ConvNeXt V2 pretraining')
    parser.add_argument('--conv_pretrain_wd', type=float, default=0.05, help='Weight decay for ConvNeXt V2 pretraining')

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
    exp_name = f"demo1_{args.dataset}_{args.model_name}_{args.optimizer}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.mkdir(os.path.join(result_path, exp_name))
    
    # Initialize TensorBoard if requested
    tb_writer = None
    if getattr(args, 'use_tensorboard', False):
        tb_log_dir = os.path.join(args.tensorboard_log_dir, exp_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        # Save configuration to logs folder
        config_save_path = os.path.join(tb_log_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"Configuration saved to: {config_save_path}")
    
    # Initialize W&B if requested
    wandb_run = None
    if getattr(args, 'use_wandb', False):
        try:
            import wandb
            run_name = args.wandb_run_name or exp_name
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                dir=os.path.join(result_path, exp_name),
            )
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")
            wandb_run = None
    if args.dataset in ['mnist', 'fashionmnist', 'cifar100', 'caltech101', 'cifar100_224']:
        dataloaders, dataset_sizes, class_names, num_classes = prepare_builtin_data(data_dir=f"data/{args.dataset}", batch_size=args.batch_size, dataset=args.dataset)
        
        # Split training data into train/val if --val is enabled
        if args.val:
            from torch.utils.data import random_split
            train_dataset = dataloaders['train'].dataset
            train_size = int((1 - args.val_size) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            dataloaders['train'] = torch.utils.data.DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            dataloaders['val'] = torch.utils.data.DataLoader(
                val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            
            dataset_sizes['train'] = len(train_subset)
            dataset_sizes['val'] = len(val_subset)
            
            print(f"\n[VALIDATION SPLIT ENABLED]")
            print(f"Original training size: {len(train_dataset)}")
            print(f"New training size: {dataset_sizes['train']}")
            print(f"Validation size: {dataset_sizes['val']} ({args.val_size*100:.1f}%)\n")
    elif args.dataset in ['intel', 'mit', 'imagenet']:
        if args.dataset == 'intel':
            train_dir = 'data/intel_image/seg_train/seg_train'
            test_dir = 'data/intel_image/seg_test/seg_test'
        elif args.dataset == 'mit':
            train_dir = 'data/mit_indoor/indoorCVPR_09/Images'
            test_dir = 'data/mit_indoor/TestImages.txt'
        elif args.dataset == 'imagenet':
            train_dir = [
                'data/imagenet/train_data_batch_1',
                # 'data/imagenet/train_data_batch_2',
                # 'data/imagenet/train_data_batch_3',
                # 'data/imagenet/train_data_batch_4',
                # 'data/imagenet/train_data_batch_5',
                # 'data/imagenet/train_data_batch_6',
                # 'data/imagenet/train_data_batch_7',
                # 'data/imagenet/train_data_batch_8',
                # 'data/imagenet/train_data_batch_9',
                # 'data/imagenet/train_data_batch_10',
            ]
            test_dir = [
                'data/imagenet/val_data',
            ]
        dataloaders, dataset_sizes, class_names, num_classes = prepare_data(train_dir= train_dir, test_dir= test_dir, input_size= args.input_size, batch_size= args.batch_size, dataset=args.dataset)
        
        # Split training data into train/val if --val is enabled
        if args.val:
            from torch.utils.data import random_split
            train_dataset = dataloaders['train'].dataset
            train_size = int((1 - args.val_size) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed)
            )
            
            dataloaders['train'] = torch.utils.data.DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4
            )
            dataloaders['val'] = torch.utils.data.DataLoader(
                val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            
            dataset_sizes['train'] = len(train_subset)
            dataset_sizes['val'] = len(val_subset)
            
            print(f"\n[VALIDATION SPLIT ENABLED]")
            print(f"Original training size: {len(train_dataset)}")
            print(f"New training size: {dataset_sizes['train']}")
            print(f"Validation size: {dataset_sizes['val']} ({args.val_size*100:.1f}%)\n")
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
        model = resnet18(num_classes = num_classes, in_channels= 3, dropout_rate=args.dropout_rate)
    elif args.model_name == 'resnet34':
        model = resnet34(num_classes = num_classes, in_channels= 3, dropout_rate=args.dropout_rate)
    elif args.model_name == 'resnet50':
        model = resnet50(num_classes= num_classes, in_channels= 3, dropout_rate=args.dropout_rate)
    elif args.model_name == 'resnet101':
        model = resnet101(num_classes= num_classes, in_channels= 3, dropout_rate=args.dropout_rate)
    elif args.model_name == 'inceptionv3':
        model = InceptionV3(num_classes=num_classes, in_channels=3)
    elif args.model_name == 'mobilenetv3_s':
        model = MobileNetV3(mode = 'small', num_classes = num_classes, dropout=args.dropout_rate)
    elif args.model_name == 'mobilenetv3_l':
        model = MobileNetV3(mode = 'large', num_classes = num_classes, dropout=args.dropout_rate)
    elif args.model_name == 'vit':
        model = VisionTransformer(num_classes = num_classes, dropout_rate= args.dropout_rate)
    elif args.model_name == 'efficientnetv2_s':
        model = EfficientNetV2(version='s', num_classes=num_classes, dropout_rate=args.dropout_rate)
    elif args.model_name == 'efficientnetv2_m':
        model = EfficientNetV2(version='m', num_classes=num_classes, dropout_rate=args.dropout_rate)
    elif args.model_name == 'efficientnetv2_l':
        model = EfficientNetV2(version='l', num_classes=num_classes, dropout_rate=args.dropout_rate)
    elif args.model_name == 'convnextv2_t':
        model = convnextv2_tiny(num_classes=num_classes, drop_path_rate=args.dropout_rate)
        
        # Self-supervised pretraining with FCMAE
        if args.conv_pretrain:
            print("\n[INFO] Starting FCMAE self-supervised pretraining for ConvNeXt V2 Tiny...")
            # Create encoder without classification head for pretraining
            encoder = convnextv2_tiny(num_classes=0, drop_path_rate=args.dropout_rate)
            pretrained_state = pretrain_convnext_fcmae(encoder, dataloaders, args, device, os.path.join(result_path, exp_name))
            # Load pretrained encoder weights into supervised model (excluding head)
            model.load_state_dict(pretrained_state, strict=False)
            print("[INFO] Pretrained encoder weights loaded into supervised model\n")
        elif args.pretrained_path:
            print(f"Loading pretrained checkpoint from {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # Remove 'head' layer from pretrained weights (will be randomly initialized)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully (head layer excluded)")
    elif args.model_name == 'convnextv2_b':
        model = convnextv2_base(num_classes=num_classes, drop_path_rate=args.dropout_rate)
        
        # Self-supervised pretraining with FCMAE
        if args.conv_pretrain:
            print("\n[INFO] Starting FCMAE self-supervised pretraining for ConvNeXt V2 Base...")
            # Create encoder without classification head for pretraining
            encoder = convnextv2_base(num_classes=0, drop_path_rate=args.dropout_rate)
            pretrained_state = pretrain_convnext_fcmae(encoder, dataloaders, args, device, os.path.join(result_path, exp_name))
            # Load pretrained encoder weights into supervised model (excluding head)
            model.load_state_dict(pretrained_state, strict=False)
            print("[INFO] Pretrained encoder weights loaded into supervised model\n")
        elif args.pretrained_path:
            print(f"Loading pretrained checkpoint from {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # Remove 'head' layer from pretrained weights (will be randomly initialized)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully (head layer excluded)")
    else:
        raise ValueError(f"Model {args.model_name} not recognized.")
    
    # ==========================================
    # RESUME FROM CHECKPOINT (if provided)
    # ==========================================
    start_epoch = 0
    loaded_history = None
    loaded_optimizer_state = None
    loaded_scheduler_state = None
    
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"{'='*60}\n")
            
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Model state loaded")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✓ Model state loaded")
            else:
                # Assume the checkpoint is just the state dict
                model.load_state_dict(checkpoint)
                print("✓ Model state loaded")
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                loaded_optimizer_state = checkpoint['optimizer_state_dict']
                print("✓ Optimizer state will be loaded")
            
            # Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                loaded_scheduler_state = checkpoint['scheduler_state_dict']
                print("✓ Scheduler state will be loaded")
            
            # Load training metadata if available
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"✓ Resuming from epoch {start_epoch}")
            
            if 'history' in checkpoint:
                loaded_history = checkpoint['history']
                print("✓ Training history loaded")
            
            print(f"\n{'='*60}\n")
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")
            print("Starting training from scratch...\n")
    
    # print(f"Model: {model}")
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not recognized.")
    
    # Load optimizer state if resuming
    if loaded_optimizer_state is not None:
        optimizer.load_state_dict(loaded_optimizer_state)
        print("✓ Optimizer state loaded")
    
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
        # Tính toán steps
        warmup_steps = args.num_warmup_steps
        total_steps = args.num_epochs * len(dataloaders['train'])

        # Decay steps chiếm TOÀN BỘ phần còn lại sau warmup
        decay_steps = total_steps - warmup_steps 

        # 1. Warmup: Tăng từ rất nhỏ lên base_lr
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, # Bắt đầu từ 1% LR
            end_factor=1.0,    # Lên 100% LR
            total_iters=warmup_steps
        )

        # 2. Decay: Giảm từ base_lr xuống min_lr
        if args.scheduler == 'cosine':
            # Eta_min thường để khá nhỏ, ví dụ 1e-6 hoặc 1% base_lr
            eta_min = args.learning_rate * 0.01 
            decay_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=decay_steps, # Decay từ từ trong suốt quãng đường còn lại
                eta_min=eta_min
            )
        elif args.scheduler == 'step':
            step_size = args.step_size * len(dataloaders['train'])  # Convert epochs to steps
            decay_scheduler = StepLR(
                optimizer,
                step_size=step_size,
                gamma=args.step_gamma
            )
        elif args.scheduler == 'onecycle':
            max_lr = args.learning_rate * args.max_lr_factor
            # OneCycleLR doesn't use warmup_scheduler, it handles warmup internally
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos'
            )
        elif args.scheduler == 'linear':
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=decay_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {args.scheduler}")

        # Nối lại: Hết warmup là sang decay luôn, không có chuyện "nghỉ giải lao" (steady)
        # Note: OneCycleLR handles warmup internally, so we don't use SequentialLR for it
        if args.scheduler != 'onecycle':
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[warmup_steps]
            )
    
    # Load scheduler state if resuming
    if loaded_scheduler_state is not None and scheduler is not None:
        scheduler.load_state_dict(loaded_scheduler_state)
        print("✓ Scheduler state loaded")
    
    # Setup Adaptive Training Config if enabled
    adaptive_config = None
    if args.adaptive_training:
        from trainer import AdaptiveTrainingConfig
        adaptive_config = AdaptiveTrainingConfig(
            enabled=True,
            check_interval=args.adaptive_check_interval
        )
        print(f"✓ Adaptive Training enabled (check interval: {args.adaptive_check_interval} epochs)")
    
    best_model_path = os.path.join(result_path, exp_name, 'best_model.pth')
    trainer = Trainer(
        model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_path=best_model_path,
        wandb_run=wandb_run,
        tb_writer=tb_writer,
        # Section 4.3: Anti-Overfitting & Adaptive Learning
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_cutmix=args.use_cutmix,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        adaptive_config=adaptive_config,
    )
    
    # Restore history and best metrics if resuming
    if loaded_history is not None:
        trainer.history = loaded_history
        if 'best_acc' in checkpoint:
            trainer.best_acc = checkpoint['best_acc']
        if 'best_val_loss' in checkpoint:
            trainer.best_val_loss = checkpoint['best_val_loss']
        print("✓ Training history and best metrics restored")
    
    if args.fasttrain:
        model, history = trainer.fasttrain()
    else:
        model, history = trainer.train()
    
    # Save final checkpoint with full state
    checkpoint_path = os.path.join(result_path, exp_name, 'checkpoint_final.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': trainer.best_acc,
        'best_val_loss': trainer.best_val_loss,
        'history': history,
        'config': vars(args),
    }, checkpoint_path)
    print(f"✓ Final checkpoint saved to {checkpoint_path}")
    
    # trainer.plot_history()
    hist_json = os.path.join(result_path, exp_name, 'training_history.json')
    hist_png = os.path.join(result_path, exp_name, 'training_history.png')
    trainer.save_history(hist_json)
    trainer.save_plot_image(hist_png)
    # Log artifacts/images to W&B
    if wandb_run is not None:
        try:
            import wandb
            if os.path.exists(hist_png):
                wandb_run.log({"plots/training_history": wandb.Image(hist_png)})
            if os.path.exists(best_model_path):
                art = wandb.Artifact(name=f"{exp_name}_best_model", type="model")
                art.add_file(best_model_path)
                wandb_run.log_artifact(art)
        except Exception as e:
            print(f"W&B logging artifacts failed: {e}")
    
    # ========== Section 6: Comprehensive Evaluation ==========
    print(f'\n{"="*60}')
    print(f'POST-TRAINING EVALUATION')
    print(f'{"="*60}\n')
    
    eval_save_path = os.path.join(result_path, exp_name)
    
    # Determine which evaluations to run
    compute_robustness = args.eval_robustness or args.eval_full
    compute_calibration = args.eval_calibration or args.eval_full
    compute_bootstrap = args.eval_bootstrap or args.eval_full
    
    if args.eval_full:
        print("Running FULL benchmark evaluation (all Section 6 metrics)...")
    elif compute_robustness or compute_calibration or compute_bootstrap:
        enabled = []
        if compute_robustness:
            enabled.append('Robustness')
        if compute_calibration:
            enabled.append('Calibration')
        if compute_bootstrap:
            enabled.append('Bootstrap CI')
        print(f"Running selective evaluation: {', '.join(enabled)}")
    else:
        print("Running basic evaluation (top-1/5 accuracy, confusion matrix)")
    
    # Run evaluation with optional comprehensive metrics
    eval_results = evaluate_model(
        model=model,
        test_dataloader=dataloaders['test'],
        num_class=num_classes,
        save_path=eval_save_path,
        compute_robustness=compute_robustness,
        compute_calibration=compute_calibration,
        compute_bootstrap_ci=compute_bootstrap,
        n_bootstrap=args.n_bootstrap,
        n_calibration_bins=args.n_calibration_bins
    )
    
    # Log comprehensive metrics to W&B and TensorBoard
    if compute_robustness and 'robustness' in eval_results:
        rob = eval_results['robustness']
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    'eval/robustness_gaussian_noise_delta': rob['gaussian_noise_delta'],
                    'eval/robustness_salt_pepper_delta': rob['salt_pepper_delta'],
                    'eval/robustness_gaussian_blur_delta': rob['gaussian_blur_delta'],
                    'eval/robustness_avg_delta': rob['avg_robustness_delta'],
                })
            except Exception as e:
                print(f"W&B robustness logging failed: {e}")
        if tb_writer is not None:
            try:
                tb_writer.add_scalar('eval/robustness_avg_delta', rob['avg_robustness_delta'], args.num_epochs)
            except Exception:
                pass
    
    if compute_calibration and 'ece' in eval_results:
        ece = eval_results['ece']
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({'eval/ece': ece})
                # Log reliability diagram if exists
                reliability_path = os.path.join(eval_save_path, 'reliability_diagram.png')
                if os.path.exists(reliability_path):
                    wandb_run.log({"eval/reliability_diagram": wandb.Image(reliability_path)})
            except Exception as e:
                print(f"W&B calibration logging failed: {e}")
        if tb_writer is not None:
            try:
                tb_writer.add_scalar('eval/ece', ece, args.num_epochs)
            except Exception:
                pass
    
    if compute_bootstrap and 'bootstrap_ci' in eval_results:
        bs = eval_results['bootstrap_ci']
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    'eval/bootstrap_mean_accuracy': bs['mean_accuracy'],
                    'eval/bootstrap_std': bs['std_accuracy'],
                    'eval/bootstrap_ci_lower': bs['lower_bound'],
                    'eval/bootstrap_ci_upper': bs['upper_bound'],
                })
            except Exception as e:
                print(f"W&B bootstrap logging failed: {e}")
        if tb_writer is not None:
            try:
                tb_writer.add_scalar('eval/bootstrap_mean_accuracy', bs['mean_accuracy'], args.num_epochs)
            except Exception:
                pass
    
    # ========== Section 2: Efficiency Benchmark ==========
    if args.eval_efficiency:
        print(f'\n{"="*60}')
        print(f'EFFICIENCY BENCHMARK (Section 2)')
        print(f'{"="*60}\n')
        
        efficiency_results = benchmark_efficiency(
            model=model,
            test_dataloader=dataloaders['test'],
            device=device,
            input_size=(3, args.input_size, args.input_size),
            batch_size=args.batch_size,
            save_path=eval_save_path
        )
        
        # Log efficiency metrics to W&B and TensorBoard
        if wandb_run is not None:
            try:
                import wandb
                wandb_run.log({
                    'efficiency/model_size_mb': efficiency_results['model_size']['size_mb'],
                    'efficiency/num_params': efficiency_results['model_size']['num_params'],
                    'efficiency/throughput_imgs_per_sec': efficiency_results['throughput']['throughput_imgs_per_sec'],
                    'efficiency/mean_latency_ms': efficiency_results['latency']['mean_latency_ms'],
                    'efficiency/p95_latency_ms': efficiency_results['latency']['p95_latency_ms'],
                })
                if 'vram' in efficiency_results and 'peak_vram_mb' in efficiency_results['vram']:
                    wandb_run.log({'efficiency/peak_vram_mb': efficiency_results['vram']['peak_vram_mb']})
            except Exception as e:
                print(f"W&B efficiency logging failed: {e}")
        
        if tb_writer is not None:
            try:
                tb_writer.add_scalar('efficiency/throughput', efficiency_results['throughput']['throughput_imgs_per_sec'], args.num_epochs)
                tb_writer.add_scalar('efficiency/latency_ms', efficiency_results['latency']['mean_latency_ms'], args.num_epochs)
            except Exception:
                pass
    
    # Close TensorBoard writer
    if tb_writer is not None:
        try:
            # Log final metrics
            tb_writer.add_hparams(
                hparam_dict={
                    'lr': args.learning_rate,
                    'batch_size': args.batch_size,
                    'optimizer': args.optimizer,
                    'model': args.model_name,
                    'dataset': args.dataset,
                },
                metric_dict={
                    'best_val_acc': float(trainer.best_acc),
                    'best_val_loss': float(trainer.best_val_loss),
                }
            )
            tb_writer.close()
            print(f"TensorBoard logs saved. View with: tensorboard --logdir={args.tensorboard_log_dir}")
        except Exception as e:
            print(f"TensorBoard closing failed: {e}")
    
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
if __name__ == "__main__":
    args = parse_args()
    main(args)