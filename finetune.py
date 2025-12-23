import os
import time
from evaluate import evaluate_model
from trainer import Trainer, count_images_per_class, calculate_class_weights
from data_preprocess import prepare_data, prepare_builtin_data
from model import *
from models_dense import convnextv2_tiny, convnextv2_base
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import json
from pathlib import Path
from typing import Any, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as tv_models


# ==========================================
# PRETRAINED WEIGHT LOADING UTILITIES
# ==========================================

def load_pretrained_resnet(custom_model, model_name='resnet18'):
    """
    Load ImageNet pretrained weights from torchvision into custom ResNet model.
    """
    print(f"Loading pretrained {model_name} weights from torchvision...")
    
    # Get pretrained model from torchvision
    if model_name == 'resnet18':
        pretrained = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet34':
        pretrained = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        pretrained = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet101':
        pretrained = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ResNet model: {model_name}")
    
    # Load state dict (excluding final FC layer)
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Filter out keys that don't match (mainly the fc layer)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # Update custom model
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


def load_pretrained_vgg16_bn(custom_model):
    """
    Load ImageNet pretrained weights from torchvision into custom VGG16_BN model.
    """
    print("Loading pretrained VGG16_BN weights from torchvision...")
    
    pretrained = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Map torchvision keys to custom model keys and filter by shape
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


def load_pretrained_efficientnetv2(custom_model, version='s'):
    """
    Load ImageNet pretrained weights from torchvision into custom EfficientNetV2 model.
    """
    print(f"Loading pretrained EfficientNetV2-{version.upper()} weights from torchvision...")
    
    if version == 's':
        pretrained = tv_models.efficientnet_v2_s(weights=tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    elif version == 'm':
        pretrained = tv_models.efficientnet_v2_m(weights=tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    elif version == 'l':
        pretrained = tv_models.efficientnet_v2_l(weights=tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported EfficientNetV2 version: {version}")
    
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Filter and map keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


def load_pretrained_mobilenetv3(custom_model, mode='large'):
    """
    Load ImageNet pretrained weights from torchvision into custom MobileNetV3 model.
    """
    print(f"Loading pretrained MobileNetV3-{mode} weights from torchvision...")
    
    if mode == 'small':
        pretrained = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    elif mode == 'large':
        pretrained = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported MobileNetV3 mode: {mode}")
    
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Filter and map keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


def load_pretrained_vit(custom_model):
    """
    Load ImageNet pretrained weights from torchvision into custom ViT model.
    """
    print("Loading pretrained ViT-B/16 weights from torchvision...")
    
    pretrained = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Filter and map keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


def load_pretrained_convnextv2(custom_model, pretrained_path, model_name='convnextv2_t'):
    """
    Load pretrained ConvNeXtV2 weights from checkpoint file or timm.
    """
    print(f"Loading pretrained {model_name} weights from checkpoint...")
    
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Filter out head/classifier if num_classes differs
        model_dict = custom_model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(state_dict)
        custom_model.load_state_dict(model_dict)
        
        print(f"✓ Loaded {len(state_dict)} pretrained layers from checkpoint")
    else:
        try:
            import timm
            if model_name == 'convnextv2_t':
                pretrained = timm.create_model('convnextv2_tiny', pretrained=True)
            elif model_name == 'convnextv2_b':
                pretrained = timm.create_model('convnextv2_base', pretrained=True)
            else:
                raise ValueError(f"Unsupported ConvNeXtV2: {model_name}")
            
            pretrained_dict = pretrained.state_dict()
            model_dict = custom_model.state_dict()
            
            # Map and filter
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(pretrained_dict)
            custom_model.load_state_dict(model_dict)
            
            print(f"✓ Loaded {len(pretrained_dict)} pretrained layers from timm")
        except ImportError:
            print("⚠️  timm not available and no checkpoint provided. Skipping pretrained weights.")
    
    return custom_model


def load_pretrained_alexnet(custom_model):
    """
    Load ImageNet pretrained weights from torchvision into custom AlexNet model.
    """
    print("Loading pretrained AlexNet weights from torchvision...")
    
    pretrained = tv_models.alexnet(weights=tv_models.AlexNet_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = custom_model.state_dict()
    
    # Filter and map keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    custom_model.load_state_dict(model_dict)
    
    print(f"✓ Loaded {len(pretrained_dict)} pretrained layers")
    return custom_model


# ==========================================
# BACKBONE FREEZING UTILITIES
# ==========================================

def freeze_backbone(model, model_name):
    """
    Freeze all layers except the final classification head.
    """
    print(f"Freezing backbone layers for {model_name}...")
    frozen_params = 0
    
    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        # Freeze all except fc layer
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name == 'alexnet':
        # Freeze features, keep classifier trainable
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name in ['vgg16', 'vgg16_bn']:
        # Freeze features, keep classifier trainable
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name in ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l']:
        # Freeze all except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name in ['mobilenetv3_s', 'mobilenetv3_l']:
        # Freeze all except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name == 'vit':
        # Freeze all except head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    elif model_name in ['convnextv2_t', 'convnextv2_b']:
        # Freeze all except head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                frozen_params += 1
    
    print(f"✓ Frozen {frozen_params} parameter groups")


def unfreeze_backbone(model):
    """
    Unfreeze all model parameters.
    """
    print("Unfreezing all backbone layers...")
    for param in model.parameters():
        param.requires_grad = True
    print("✓ All layers unfrozen")


def get_trainable_params(model):
    """
    Get number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# ARGUMENT PARSING
# ==========================================

def parse_finetune_args(input_args=None):
    parser = argparse.ArgumentParser(description="Fine-tuning script with pretrained ImageNet weights")
    
    # Dataset & Data
    parser.add_argument('--dataset', type=str, default='mit', 
                        choices=['mnist', 'intel', 'fashionmnist', 'cifar100', 'mit', 'imagenet', 'caltech101'],
                        help='Dataset to use for fine-tuning')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    
    # Model
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l', 
                                'alexnet', 'vgg16_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                'mobilenetv3_s', 'mobilenetv3_l', 'vit', 
                                'convnextv2_t', 'convnextv2_b'],
                        help='Name of the model to use')
    parser.add_argument('--pretrained_path', type=str, default=None, 
                        help='Path to pretrained checkpoint (for ConvNeXtV2)')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate for model')
    
    # Fine-tuning Strategy
    parser.add_argument('--freeze_epochs', type=int, default=5, 
                        help='Number of epochs to train with frozen backbone (0 = no freezing)')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, 
                        help='Learning rate for backbone (after unfreezing)')
    parser.add_argument('--head_lr', type=float, default=1e-3, 
                        help='Learning rate for classification head')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=50, help='Total number of epochs')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['constant', 'linear', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--num_warmup_steps', type=int, default=500, help='Number of warmup steps')
    
    # Loss & Regularization
    parser.add_argument('--criterion', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'mse', 'hinge'], help='Loss function to use')
    parser.add_argument('--use_class_weights', action='store_true', 
                        help='Use class weights for loss function')
    parser.add_argument('--weight_type', type=str, default='inverse', 
                        choices=['inverse', 'sqrt_inverse'], help='Type of class weights to use')
    
    # Logging & Saving
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='dl20251-finetune', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team or username)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_log_dir', type=str, default='logs', help='TensorBoard log directory')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--fasttrain', action='store_true', help='Enable fast training with mixed precision (torch.cuda.amp)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    args = parser.parse_args(input_args)
    return args


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main(args):
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results directory
    result_path = os.path.abspath('results_finetune')
    os.makedirs(result_path, exist_ok=True)
    exp_name = f"ft_{args.dataset}_{args.model_name}_{args.optimizer}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(os.path.join(result_path, exp_name), exist_ok=True)
    
    # Initialize TensorBoard
    tb_writer = None
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.tensorboard_log_dir, exp_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        config_save_path = os.path.join(tb_log_dir, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"TensorBoard logs: {tb_log_dir}")
    
    # Initialize W&B
    wandb_run = None
    if args.use_wandb:
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
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {args.dataset} dataset...")
    print(f"{'='*60}\n")
    
    if args.dataset in ['mnist', 'fashionmnist', 'cifar100', 'caltech101']:
        dataloaders, dataset_sizes, class_names, num_classes = prepare_builtin_data(
            data_dir=f"data/{args.dataset}", 
            batch_size=args.batch_size, 
            dataset=args.dataset
        )
    elif args.dataset in ['intel', 'mit', 'imagenet']:
        if args.dataset == 'intel':
            train_dir = 'data/intel_image/seg_train/seg_train'
            test_dir = 'data/intel_image/seg_test/seg_test'
        elif args.dataset == 'mit':
            train_dir = 'data/mit_indoor/indoorCVPR_09/Images'
            test_dir = 'data/mit_indoor/TestImages.txt'
        elif args.dataset == 'imagenet':
            train_dir = ['data/imagenet/train_data_batch_1']
            test_dir = ['data/imagenet/val_data']
        
        dataloaders, dataset_sizes, class_names, num_classes = prepare_data(
            train_dir=train_dir, 
            test_dir=test_dir, 
            input_size=args.input_size, 
            batch_size=args.batch_size, 
            dataset=args.dataset
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    print(f"\n{'='*60}")
    print(f"Initializing {args.model_name} model...")
    print(f"{'='*60}\n")
    
    if args.model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
        model = load_pretrained_alexnet(model)
    elif args.model_name == 'vgg16_bn':
        model = VGG16BatchNorm(num_classes=num_classes, in_channels=3, 
                               dropout_rate=args.dropout_rate, input_size=args.input_size)
        model = load_pretrained_vgg16_bn(model)
    elif args.model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, in_channels=3)
        model = load_pretrained_resnet(model, 'resnet18')
    elif args.model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, in_channels=3)
        model = load_pretrained_resnet(model, 'resnet34')
    elif args.model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, in_channels=3)
        model = load_pretrained_resnet(model, 'resnet50')
    elif args.model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, in_channels=3)
        model = load_pretrained_resnet(model, 'resnet101')
    elif args.model_name == 'mobilenetv3_s':
        model = MobileNetV3(mode='small', num_classes=num_classes, dropout=args.dropout_rate)
        model = load_pretrained_mobilenetv3(model, 'small')
    elif args.model_name == 'mobilenetv3_l':
        model = MobileNetV3(mode='large', num_classes=num_classes, dropout=args.dropout_rate)
        model = load_pretrained_mobilenetv3(model, 'large')
    elif args.model_name == 'vit':
        model = VisionTransformer(num_classes=num_classes, dropout_rate=args.dropout_rate)
        model = load_pretrained_vit(model)
    elif args.model_name == 'efficientnetv2_s':
        model = EfficientNetV2(version='s', num_classes=num_classes, dropout_rate=args.dropout_rate)
        model = load_pretrained_efficientnetv2(model, 's')
    elif args.model_name == 'efficientnetv2_m':
        model = EfficientNetV2(version='m', num_classes=num_classes, dropout_rate=args.dropout_rate)
        model = load_pretrained_efficientnetv2(model, 'm')
    elif args.model_name == 'efficientnetv2_l':
        model = EfficientNetV2(version='l', num_classes=num_classes, dropout_rate=args.dropout_rate)
        model = load_pretrained_efficientnetv2(model, 'l')
    elif args.model_name == 'convnextv2_t':
        model = convnextv2_tiny(num_classes=num_classes)
        model = load_pretrained_convnextv2(model, args.pretrained_path, 'convnextv2_t')
    elif args.model_name == 'convnextv2_b':
        model = convnextv2_base(num_classes=num_classes)
        model = load_pretrained_convnextv2(model, args.pretrained_path, 'convnextv2_b')
    else:
        raise ValueError(f"Model {args.model_name} not recognized.")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = get_trainable_params(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==========================================
    # RESUME FROM CHECKPOINT (if provided)
    # ==========================================
    start_epoch = 0
    resume_phase = None
    loaded_history = None
    
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {args.resume}")
            print(f"{'='*60}\n")
            
            checkpoint = torch.load(args.resume, map_location=device)
            
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
            
            # Load training metadata if available
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"✓ Resuming from epoch {start_epoch}")
            
            if 'phase' in checkpoint:
                resume_phase = checkpoint['phase']
                print(f"✓ Resuming from phase: {resume_phase}")
            
            if 'history' in checkpoint:
                loaded_history = checkpoint['history']
                print("✓ Training history loaded")
            
            print(f"\n{'='*60}\n")
        else:
            print(f"Warning: Checkpoint file not found: {args.resume}")
            print("Starting training from scratch...\n")
    
    # Setup optimizer with differential learning rates
    if args.freeze_epochs > 0:
        # Initial phase: only train head
        freeze_backbone(model, args.model_name)
        trainable_params = get_trainable_params(model)
        print(f"\n📌 PHASE 1: Training only classification head")
        print(f"Trainable parameters: {trainable_params:,}")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=args.head_lr)
    else:
        # Train all with differential LR
        print(f"\n📌 Training full model with differential learning rates")
        print(f"Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")
        
        # Separate backbone and head parameters
        if args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            head_params = list(model.fc.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        elif args.model_name in ['alexnet', 'vgg16', 'vgg16_bn']:
            head_params = list(model.classifier.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        elif args.model_name in ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
                                  'mobilenetv3_s', 'mobilenetv3_l']:
            head_params = list(model.classifier.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        elif args.model_name == 'vit':
            head_params = list(model.head.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        elif args.model_name in ['convnextv2_t', 'convnextv2_b']:
            head_params = list(model.head.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        else:
            head_params = []
            backbone_params = list(model.parameters())
        
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': head_params, 'lr': args.head_lr}
            ])
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD([
                {'params': backbone_params, 'lr': args.backbone_lr, 'momentum': 0.9},
                {'params': head_params, 'lr': args.head_lr, 'momentum': 0.9}
            ])
        elif args.optimizer == 'adam':
            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': head_params, 'lr': args.head_lr}
            ])
    
    # Setup criterion
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
    
    # Setup scheduler (only for frozen phase or full training)
    if args.scheduler == 'constant':
        scheduler = None
    else:
        total_steps = args.num_epochs * len(dataloaders['train'])
        warmup_steps = args.num_warmup_steps
        decay_steps = total_steps - warmup_steps
        
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                                     total_iters=warmup_steps)
        
        if args.scheduler == 'cosine':
            eta_min = args.backbone_lr * 0.01
            decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
        elif args.scheduler == 'linear':
            decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, 
                                       total_iters=decay_steps)
        
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                                milestones=[warmup_steps])
    
    # ==========================================
    # PHASE 1: Train with frozen backbone
    # ==========================================
    if args.freeze_epochs > 0 and (resume_phase is None or resume_phase == 'phase1'):
        print(f"\n{'='*60}")
        print(f"PHASE 1: Training with frozen backbone ({args.freeze_epochs} epochs)")
        if start_epoch > 0 and resume_phase == 'phase1':
            print(f"Resuming from epoch {start_epoch + 1}")
        print(f"{'='*60}\n")
        
        best_model_path = os.path.join(result_path, exp_name, 'best_model_phase1.pth')
        trainer = Trainer(
            model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.freeze_epochs,
            save_path=best_model_path,
            wandb_run=wandb_run,
            tb_writer=tb_writer,
        )
        
        # Restore history if resuming
        if loaded_history is not None and resume_phase == 'phase1':
            trainer.history = loaded_history
            trainer.best_acc = checkpoint.get('best_acc', 0.0)
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if args.fasttrain:
            model, history_phase1 = trainer.fasttrain()
        else:
            model, history_phase1 = trainer.train()
        
        # Save phase 1 checkpoint with full state
        checkpoint_path = os.path.join(result_path, exp_name, 'checkpoint_phase1.pth')
        torch.save({
            'epoch': args.freeze_epochs,
            'phase': 'phase1',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_acc': trainer.best_acc,
            'best_val_loss': trainer.best_val_loss,
            'history': history_phase1,
        }, checkpoint_path)
        print(f"✓ Phase 1 checkpoint saved to {checkpoint_path}")
        
        # Save phase 1 history
        hist_json = os.path.join(result_path, exp_name, 'training_history_phase1.json')
        hist_png = os.path.join(result_path, exp_name, 'training_history_phase1.png')
        trainer.save_history(hist_json)
        trainer.save_plot_image(hist_png)
    elif args.freeze_epochs > 0 and resume_phase == 'phase2':
        print(f"\n{'='*60}")
        print("Skipping Phase 1 (already completed in checkpoint)")
        print(f"{'='*60}\n")
    
    # ==========================================
    # PHASE 2: Unfreeze and fine-tune full model
    # ==========================================
    remaining_epochs = args.num_epochs - args.freeze_epochs
    phase2_start_epoch = 0
    
    # Determine starting point for phase 2
    if resume_phase == 'phase2' and start_epoch > args.freeze_epochs:
        phase2_start_epoch = start_epoch - args.freeze_epochs
        print(f"Resuming Phase 2 from epoch {phase2_start_epoch + 1}/{remaining_epochs}")
    
    if remaining_epochs > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Fine-tuning full model ({remaining_epochs} epochs)")
        print(f"{'='*60}\n")
        
        # Unfreeze all layers
        unfreeze_backbone(model)
        trainable_params = get_trainable_params(model)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create new optimizer with differential learning rates
        if args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            head_params = list(model.fc.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        elif args.model_name in ['alexnet', 'vgg16', 'vgg16_bn']:
            head_params = list(model.classifier.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        elif args.model_name in ['efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
                                  'mobilenetv3_s', 'mobilenetv3_l']:
            head_params = list(model.classifier.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
        elif args.model_name == 'vit':
            head_params = list(model.head.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        elif args.model_name in ['convnextv2_t', 'convnextv2_b']:
            head_params = list(model.head.parameters())
            backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
        else:
            head_params = []
            backbone_params = list(model.parameters())
        
        if args.optimizer == 'adamw':
            optimizer_phase2 = optim.AdamW([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': head_params, 'lr': args.head_lr}
            ])
        elif args.optimizer == 'sgd':
            optimizer_phase2 = optim.SGD([
                {'params': backbone_params, 'lr': args.backbone_lr, 'momentum': 0.9},
                {'params': head_params, 'lr': args.head_lr, 'momentum': 0.9}
            ])
        elif args.optimizer == 'adam':
            optimizer_phase2 = optim.Adam([
                {'params': backbone_params, 'lr': args.backbone_lr},
                {'params': head_params, 'lr': args.head_lr}
            ])
        
        # Setup scheduler for phase 2
        if args.scheduler == 'constant':
            scheduler_phase2 = None
        else:
            total_steps_p2 = remaining_epochs * len(dataloaders['train'])
            warmup_steps_p2 = args.num_warmup_steps
            decay_steps_p2 = total_steps_p2 - warmup_steps_p2
            
            warmup_scheduler_p2 = LinearLR(optimizer_phase2, start_factor=0.01, 
                                           end_factor=1.0, total_iters=warmup_steps_p2)
            
            if args.scheduler == 'cosine':
                eta_min_p2 = args.backbone_lr * 0.01
                decay_scheduler_p2 = CosineAnnealingLR(optimizer_phase2, T_max=decay_steps_p2, 
                                                        eta_min=eta_min_p2)
            elif args.scheduler == 'linear':
                decay_scheduler_p2 = LinearLR(optimizer_phase2, start_factor=1.0, 
                                              end_factor=0.01, total_iters=decay_steps_p2)
            
            scheduler_phase2 = SequentialLR(optimizer_phase2, 
                                           schedulers=[warmup_scheduler_p2, decay_scheduler_p2],
                                           milestones=[warmup_steps_p2])
        
        best_model_path = os.path.join(result_path, exp_name, 'best_model.pth')
        trainer_phase2 = Trainer(
            model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            criterion=criterion,
            optimizer=optimizer_phase2,
            scheduler=scheduler_phase2,
            device=device,
            num_epochs=remaining_epochs,
            save_path=best_model_path,
            wandb_run=wandb_run,
            tb_writer=tb_writer,
        )
        
        # Restore history if resuming phase 2
        if loaded_history is not None and resume_phase == 'phase2':
            trainer_phase2.history = loaded_history
            trainer_phase2.best_acc = checkpoint.get('best_acc', 0.0)
            trainer_phase2.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if args.fasttrain:
            model, history_phase2 = trainer_phase2.fasttrain()
        else:
            model, history_phase2 = trainer_phase2.train()
        
        # Save phase 2 checkpoint with full state
        checkpoint_path = os.path.join(result_path, exp_name, 'checkpoint_phase2.pth')
        torch.save({
            'epoch': args.num_epochs,
            'phase': 'phase2',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_phase2.state_dict(),
            'scheduler_state_dict': scheduler_phase2.state_dict() if scheduler_phase2 else None,
            'best_acc': trainer_phase2.best_acc,
            'best_val_loss': trainer_phase2.best_val_loss,
            'history': history_phase2,
        }, checkpoint_path)
        print(f"✓ Phase 2 checkpoint saved to {checkpoint_path}")
        
        # Save phase 2 history
        hist_json = os.path.join(result_path, exp_name, 'training_history_phase2.json')
        hist_png = os.path.join(result_path, exp_name, 'training_history_phase2.png')
        trainer_phase2.save_history(hist_json)
        trainer_phase2.save_plot_image(hist_png)
    
    # ==========================================
    # EVALUATION
    # ==========================================
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}\n")
    
    evaluate_model(model, dataloaders['test'], num_class=num_classes, 
                   save_path=os.path.join(result_path, exp_name))
    
    # Close loggers
    if tb_writer is not None:
        try:
            if args.freeze_epochs > 0:
                best_acc = max(trainer.best_acc, trainer_phase2.best_acc)
                best_loss = min(trainer.best_val_loss, trainer_phase2.best_val_loss)
            else:
                best_acc = trainer_phase2.best_acc
                best_loss = trainer_phase2.best_val_loss
            
            tb_writer.add_hparams(
                hparam_dict={
                    'backbone_lr': args.backbone_lr,
                    'head_lr': args.head_lr,
                    'freeze_epochs': args.freeze_epochs,
                    'batch_size': args.batch_size,
                    'optimizer': args.optimizer,
                    'model': args.model_name,
                    'dataset': args.dataset,
                },
                metric_dict={
                    'best_val_acc': float(best_acc),
                    'best_val_loss': float(best_loss),
                }
            )
            tb_writer.close()
        except Exception as e:
            print(f"TensorBoard closing failed: {e}")
    
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    
    print(f"\n{'='*60}")
    print(f"✓ Fine-tuning completed!")
    print(f"Results saved to: {os.path.join(result_path, exp_name)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_finetune_args()
    main(args)
