import os
import time
import torch
import torch.optim as optim
import argparse
import random
import json
from pathlib import Path
import numpy as np
from data_preprocess import prepare_data, prepare_builtin_data

# --- Import module Dense ---
try:
    import models_dense as dense_models
except ImportError as e:
    exit(1)

def parse_pretrain_args():
    parser = argparse.ArgumentParser(description="ConvNeXt V2 FCMAE Pre-training Script")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='intel',
                        choices=['mnist', 'intel', 'fashionmnist', 'cifar100', 'mit', 'imagenet', 'caltech101'],
                        help='Dataset to use for pretraining')
    
    # Data & Paths
    parser.add_argument('--train_dir', type=str, default=None, help='Path to directory containing training images (optional, auto-determined for built-in datasets)')
    parser.add_argument('--output_dir', type=str, default='results_pretrain', help='Directory to save checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    
    # Model
    parser.add_argument('--model_name', type=str, default='fcmae_convnextv2_base', 
                        choices=['fcmae_convnextv2_base'],
                        help='Which FCMAE variant to use')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--mask_ratio', type=float, default=0.6, help='Masking ratio for MAE')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--learning_rate', type=float, default=1.5e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers')
    parser.add_argument('--save_freq', type=int, default=10, help='Save freq')

    args = parser.parse_args()
    return args

def save_encoder_checkpoint(model, optimizer, epoch, save_path, config_info=None):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    encoder_state = {}
    for k, v in model_state.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.', '')
            encoder_state[new_key] = v

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': encoder_state, 
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config_info
    }
    torch.save(checkpoint, save_path)
    print(f"--> Saved CLEAN ENCODER checkpoint to: {save_path}")

def pretrain_one_epoch(model, dataloader, optimizer, device, epoch, mask_ratio, print_freq=50):
    model.train()
    running_loss = 0.0
    header = f'Epoch: [{epoch+1}]'
    start_time = time.time()
    
    for i, data in enumerate(dataloader):
        if isinstance(data, (tuple, list)):
             images = data[0]
        else:
             images = data

        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, _, _ = model(images, mask_ratio=mask_ratio)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            print(f'[{header}, Batch {i+1}/{len(dataloader)}] Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s')
            running_loss = 0.0
            start_time = time.time()

    return running_loss / len(dataloader) if len(dataloader) > 0 else 0

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting FCMAE Pre-training on device: {device}")

    if args.experiment_name:
         exp_name = args.experiment_name
    else:
         exp_name = f"{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Results will be saved to: {exp_dir}")

    # 2. Data Preparation
    print("\n--> Preparing Data...")
    if args.dataset in ['mnist', 'fashionmnist', 'cifar100', 'caltech101']:
        dataloaders, dataset_sizes, class_names, num_classes = prepare_builtin_data(data_dir=f"data/{args.dataset}", batch_size=args.batch_size, dataset=args.dataset)
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
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")
    
    train_loader = dataloaders['train']
    print(f"Training samples: {dataset_sizes['train']}")
    print(f"Class names: {class_names}")

    # 3. Model Initialization
    print(f"\n--> Initializing Model: {args.model_name}")
    
    if args.model_name == 'fcmae_convnextv2_base':
        encoder = dense_models.convnextv2_base(num_classes=0)
        model = dense_models.FCMAE_Dense(encoder=encoder, mask_ratio=args.mask_ratio)
    else:
        raise ValueError(f"Model {args.model_name} is not supported in models_dense.py yet.")
    
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs - args.warmup_epochs, 
        eta_min=1e-6
    )

    # 5. Training Loop
    print(f"\n--> Starting Pre-training for {args.num_epochs} epochs...")
    global_start_time = time.time()
    
    for epoch in range(args.num_epochs):
        if epoch < args.warmup_epochs:
            lr_scale = min(1., float(epoch + 1) / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate * lr_scale
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nStart of Epoch {epoch+1}/{args.num_epochs} (LR: {current_lr:.6f})")
        
        epoch_loss = pretrain_one_epoch(model, train_loader, optimizer, device, epoch, args.mask_ratio)
        
        if epoch >= args.warmup_epochs:
            scheduler.step()
        
        print(f"End of Epoch {epoch+1}. Average Loss: {epoch_loss:.4f}")

        is_checkpoint_epoch = (epoch + 1) % args.save_freq == 0
        is_last_epoch = (epoch + 1) == args.num_epochs

        if is_checkpoint_epoch or is_last_epoch:
            save_name = f'checkpoint_encoder_ep{epoch+1}.pth' if not is_last_epoch else 'checkpoint_encoder_final.pth'
            save_path = os.path.join(exp_dir, save_name)
            save_encoder_checkpoint(model, optimizer, epoch, save_path, config_info=vars(args))

    total_time = time.time() - global_start_time
    print(f"\n--> Pre-training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Final weights saved in: {exp_dir}")

if __name__ == "__main__":
    args = parse_pretrain_args()
    main(args)