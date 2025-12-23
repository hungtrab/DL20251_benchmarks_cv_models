import os
import re
import torch
import argparse
from pathlib import Path
from datetime import datetime
from model import (
    AlexNet, LeNet, VGG16, VGG16BatchNorm,
    resnet18, resnet34, resnet50, resnet101,
    InceptionV3, MobileNetV3, VisionTransformer,
    EfficientNetV2
)
from data_preprocess import prepare_data, prepare_builtin_data
from evaluate import evaluate_model


# Dataset configurations
DATASET_CONFIG = {
    'mnist': {
        'num_classes': 10,
        'in_channels': 1,
        'input_size': 224,
        'data_loader': lambda: prepare_builtin_data(data_dir='data/mnist', batch_size=32, dataset='mnist')
    },
    'fashionmnist': {
        'num_classes': 10,
        'in_channels': 1,
        'input_size': 224,
        'data_loader': lambda: prepare_builtin_data(data_dir='data/fashionmnist', batch_size=32, dataset='fashionmnist')
    },
    'cifar100': {
        'num_classes': 100,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_builtin_data(data_dir='data/cifar100', batch_size=32, dataset='cifar100')
    },
    'cifar100_224': {
        'num_classes': 100,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_builtin_data(data_dir='data/cifar100_224', batch_size=32, dataset='cifar100_224')
    },
    'caltech101': {
        'num_classes': 101,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_builtin_data(data_dir='data/caltech101', batch_size=32, dataset='caltech101')
    },
    'intel': {
        'num_classes': 6,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_data(
            train_dir='data/intel_image/seg_train/seg_train',
            test_dir='data/intel_image/seg_test/seg_test',
            input_size=224,
            batch_size=32,
            dataset='intel'
        )
    },
    'mit': {
        'num_classes': 67,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_data(
            train_dir='data/mit_indoor/indoorCVPR_09/Images',
            test_dir='data/mit_indoor/TestImages.txt',
            input_size=224,
            batch_size=32,
            dataset='mit'
        )
    },
    'imagenet': {
        'num_classes': 1000,
        'in_channels': 3,
        'input_size': 224,
        'data_loader': lambda: prepare_data(
            train_dir=['data/imagenet/train_data_batch_1'],
            test_dir=['data/imagenet/val_data'],
            input_size=224,
            batch_size=32,
            dataset='imagenet'
        )
    }
}


def get_model_instance(model_name, num_classes, in_channels=3, input_size=224, dropout_rate=0.4):
    """
    Load model architecture matching train.py logic.
    """
    if model_name == 'alexnet':
        model = AlexNet(num_classes=num_classes)
    elif model_name == 'lenet':
        model = LeNet(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'vgg16':
        model = VGG16(num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, input_size=input_size)
    elif model_name == 'vgg16_bn':
        model = VGG16BatchNorm(num_classes=num_classes, in_channels=in_channels, dropout_rate=dropout_rate, input_size=input_size)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'inceptionv3':
        model = InceptionV3(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'mobilenetv3_s':
        model = MobileNetV3(mode='small', num_classes=num_classes, dropout=dropout_rate)
    elif model_name == 'mobilenetv3_l':
        model = MobileNetV3(mode='large', num_classes=num_classes, dropout=dropout_rate)
    elif model_name == 'vit':
        model = VisionTransformer(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'efficientnetv2_s':
        model = EfficientNetV2(version='s', num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'efficientnetv2_m':
        model = EfficientNetV2(version='m', num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'efficientnetv2_l':
        model = EfficientNetV2(version='l', num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model


def parse_run_name(run_name):
    """
    Parse run name to extract dataset, model_name, optimizer, and timestamp.
    Expected format: {dataset}_{model_name}_{optimizer}_{timestamp}
    
    Returns:
        tuple: (dataset, model_name, optimizer, timestamp) or None if parsing fails
    """
    # Pattern: dataset_modelname_optimizer_timestamp
    # Timestamp format: YYYYMMDD_HHMMSS
    pattern = r'^(.+?)_([a-z0-9_]+)_([a-z]+)_(\d{8}_\d{6})$'
    match = re.match(pattern, run_name)
    
    if match:
        dataset, model_name, optimizer, timestamp = match.groups()
        # Validate dataset
        if dataset not in DATASET_CONFIG:
            return None
        # Validate model_name
        valid_models = [
            'alexnet', 'lenet', 'vgg16', 'vgg16_bn',
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'inceptionv3', 'mobilenetv3_s', 'mobilenetv3_l',
            'vit', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
        ]
        if model_name not in valid_models:
            return None
        return dataset, model_name, optimizer, timestamp
    return None


def evaluate_run(run_path, run_name, device='cuda'):
    """
    Evaluate a single model run and save results to text file.
    """
    # Parse run name
    parsed = parse_run_name(run_name)
    if parsed is None:
        print(f"⚠️  Skipping {run_name}: Invalid format")
        return None
    
    dataset, model_name, optimizer, timestamp = parsed
    print(f"\n{'='*80}")
    print(f"Evaluating: {run_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Model: {model_name}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Timestamp: {timestamp}")
    print(f"{'='*80}")
    
    # Check if best_model.pth exists
    model_path = os.path.join(run_path, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Get dataset config
    dataset_config = DATASET_CONFIG[dataset]
    num_classes = dataset_config['num_classes']
    in_channels = dataset_config['in_channels']
    input_size = dataset_config['input_size']
    
    try:
        # Load data
        print(f"Loading {dataset} dataset...")
        dataloaders, dataset_sizes, class_names, _ = dataset_config['data_loader']()
        
        # Initialize model
        print(f"Initializing {model_name} model...")
        model = get_model_instance(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            dropout_rate=0.4
        )
        
        # Load weights
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        # Evaluate
        print(f"Evaluating on test set...")
        results = evaluate_model(
            model=model,
            test_dataloader=dataloaders['test'],
            num_class=num_classes,
            save_path=run_path
        )
        
        # Save results to text file
        txt_path = os.path.join(run_path, 'evaluation_results.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Evaluation Results for {run_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Dataset: {dataset}\n")
            f.write(f"  Model: {model_name}\n")
            f.write(f"  Optimizer: {optimizer}\n")
            f.write(f"  Timestamp: {timestamp}\n")
            f.write(f"  Number of Classes: {num_classes}\n")
            f.write(f"  Test Set Size: {dataset_sizes['test']}\n")
            f.write(f"\n{'='*80}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Top-1 Accuracy: {results['top1_accuracy']:.2f}%\n")
            f.write(f"  Top-1 Error: {results['top1_error']:.2f}%\n")
            if results['top5_accuracy'] is not None:
                f.write(f"  Top-5 Accuracy: {results['top5_accuracy']:.2f}%\n")
                f.write(f"  Top-5 Error: {results['top5_error']:.2f}%\n")
            f.write(f"  Evaluation Time: {results['eval_time']:.2f} seconds\n")
            f.write(f"\n{'='*80}\n\n")
            f.write(f"Classification Report:\n")
            f.write(results['classification_report'])
        
        print(f"✅ Results saved to {txt_path}")
        return results
        
    except Exception as e:
        import traceback
        print(f"❌ Error evaluating {run_name}: {str(e)}")
        traceback.print_exc()
        
        # Save error to file
        error_path = os.path.join(run_path, 'evaluation_error.txt')
        with open(error_path, 'w') as f:
            f.write(f"Error evaluating {run_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models in results folder")
    parser.add_argument('--results_dir', type=str, default='results', help='Path to results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for evaluation')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='Specific run name to evaluate (optional)')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"🔍 Scanning results directory: {results_dir}")
    print(f"📱 Using device: {args.device}")
    
    # Get all subdirectories
    if args.run_name:
        # Evaluate specific run
        run_path = results_dir / args.run_name
        if not run_path.exists():
            print(f"❌ Run not found: {run_path}")
            return
        run_dirs = [(args.run_name, run_path)]
    else:
        # Evaluate all runs
        run_dirs = [(d.name, d) for d in results_dir.iterdir() if d.is_dir()]
    
    print(f"📂 Found {len(run_dirs)} directories\n")
    
    # Statistics
    total_runs = len(run_dirs)
    successful = 0
    skipped = 0
    failed = 0
    
    # Evaluate each run
    for run_name, run_path in sorted(run_dirs):
        result = evaluate_run(run_path, run_name, device=args.device)
        if result is not None:
            successful += 1
        elif parse_run_name(run_name) is None:
            skipped += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total directories: {total_runs}")
    print(f"✅ Successfully evaluated: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped (invalid format): {skipped}")
    print(f"{'='*80}\n")
    
    # Create summary file
    summary_path = results_dir / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total directories: {total_runs}\n")
        f.write(f"Successfully evaluated: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Skipped (invalid format): {skipped}\n")
    
    print(f"📄 Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
