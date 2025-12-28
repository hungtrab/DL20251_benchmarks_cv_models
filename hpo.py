#!/usr/bin/env python3
"""
Hyperparameter Optimization (HPO) Module using Optuna.

This module implements Bayesian Optimization for finding optimal hyperparameters
on CIFAR100 as a proxy dataset, then transfers the best parameters to other datasets.

Strategy:
1. Group-Based HPO: 70 trials per model group (Legacy, ResNet, Modern) on CIFAR100
2. Per-Model Fine-Tuning: 10-15 additional trials per individual model starting from group's best params

Features:
- SQLite storage for persistence and resumability
- MedianPruner for efficient early stopping
- Parallel execution support via Optuna's distributed optimization
- Automatic config generation for best hyperparameters
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, StepLR, OneCycleLR
)

try:
    import optuna
    from optuna.trial import Trial
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run 'pip install optuna' to enable HPO.")

from trainer import Trainer, LabelSmoothingCrossEntropy, count_images_per_class, calculate_class_weights
from data_preprocess import prepare_data, prepare_builtin_data
from model import (
    AlexNet, LeNet, VGG16, VGG16BatchNorm,
    resnet18, resnet34, resnet50, resnet101,
    MobileNetV3, EfficientNetV2, VisionTransformer
)
from models_dense import convnextv2_tiny, convnextv2_base


# ===================== Configuration =====================

@dataclass
class HPOConfig:
    """Configuration for HPO runs."""
    # Dataset settings
    dataset: str = 'cifar100'
    input_size: int = 224
    batch_size: int = 32
    
    # HPO settings
    n_trials: int = 70
    n_finetune_trials: int = 15
    n_epochs_per_trial: int = 25  # Short trials for HPO
    n_warmup_steps: int = 500
    
    # Optuna settings
    study_name: str = 'hpo_study'
    storage_path: str = 'hpo_results.db'
    n_startup_trials: int = 5  # Number of random trials before pruning
    n_warmup_steps_pruner: int = 10  # Epochs before pruning starts
    
    # Parallel execution
    n_jobs: int = 1  # Number of parallel trials (-1 for auto)
    
    # Reproducibility
    seed: int = 42
    
    # Output
    output_dir: str = 'config/hpo'


# ===================== Model Groups =====================

MODEL_GROUPS = {
    'legacy': ['alexnet', 'vgg16_bn'],
    'resnet': ['resnet18', 'resnet34', 'resnet50'],
    'modern': ['mobilenetv3_l', 'efficientnetv2_s', 'convnextv2_t']
}

ALL_MODELS = [m for group in MODEL_GROUPS.values() for m in group]


# ===================== Search Space =====================

def get_search_space(trial: 'Trial', finetune_from: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Define the hyperparameter search space.
    
    Args:
        trial: Optuna trial object
        finetune_from: Best params from group HPO (for narrowed search in fine-tuning)
    
    Returns:
        Dictionary of hyperparameters
    """
    if finetune_from is not None:
        # Fine-tuning: narrowed search range (±20% around best values)
        base_lr = finetune_from['learning_rate']
        base_wd = finetune_from['weight_decay']
        base_dropout = finetune_from['dropout_rate']
        base_label_smooth = finetune_from['label_smoothing']
        
        params = {
            'optimizer': trial.suggest_categorical('optimizer', [finetune_from['optimizer']]),
            'learning_rate': trial.suggest_float('learning_rate', base_lr * 0.8, base_lr * 1.2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', base_wd * 0.8, base_wd * 1.2, log=True),
            'scheduler': trial.suggest_categorical('scheduler', [finetune_from['scheduler']]),
            'dropout_rate': trial.suggest_float('dropout_rate', max(0.0, base_dropout - 0.1), min(0.5, base_dropout + 0.1)),
            'label_smoothing': trial.suggest_float('label_smoothing', max(0.0, base_label_smooth - 0.05), min(0.2, base_label_smooth + 0.05)),
        }
    else:
        # Full search space
        params = {
            'optimizer': trial.suggest_categorical('optimizer', ['sgd', 'adamw']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'onecycle', 'step']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        }
    
    # Scheduler-specific parameters
    if params['scheduler'] == 'step':
        params['step_size'] = trial.suggest_int('step_size', 5, 15)
        params['step_gamma'] = trial.suggest_float('step_gamma', 0.1, 0.5)
    elif params['scheduler'] == 'onecycle':
        params['max_lr_factor'] = trial.suggest_float('max_lr_factor', 5.0, 15.0)
    
    return params


# ===================== Model Factory =====================

def create_model(model_name: str, num_classes: int, dropout_rate: float = 0.4) -> nn.Module:
    """Create a model instance by name."""
    model_map = {
        'lenet': lambda: LeNet(num_classes=num_classes, in_channels=3),
        'alexnet': lambda: AlexNet(num_classes=num_classes),
        'vgg16': lambda: VGG16(num_classes=num_classes, in_channels=3, dropout_rate=dropout_rate),
        'vgg16_bn': lambda: VGG16BatchNorm(num_classes=num_classes, in_channels=3, dropout_rate=dropout_rate),
        'resnet18': lambda: resnet18(num_classes=num_classes, in_channels=3),
        'resnet34': lambda: resnet34(num_classes=num_classes, in_channels=3),
        'resnet50': lambda: resnet50(num_classes=num_classes, in_channels=3),
        'resnet101': lambda: resnet101(num_classes=num_classes, in_channels=3),
        'mobilenetv3_s': lambda: MobileNetV3(mode='small', num_classes=num_classes, dropout=dropout_rate),
        'mobilenetv3_l': lambda: MobileNetV3(mode='large', num_classes=num_classes, dropout=dropout_rate),
        'efficientnetv2_s': lambda: EfficientNetV2(version='s', num_classes=num_classes, dropout_rate=dropout_rate),
        'efficientnetv2_m': lambda: EfficientNetV2(version='m', num_classes=num_classes, dropout_rate=dropout_rate),
        'efficientnetv2_l': lambda: EfficientNetV2(version='l', num_classes=num_classes, dropout_rate=dropout_rate),
        'vit': lambda: VisionTransformer(num_classes=num_classes, dropout_rate=dropout_rate),
        'convnextv2_t': lambda: convnextv2_tiny(num_classes=num_classes, drop_path_rate=dropout_rate),
        'convnextv2_b': lambda: convnextv2_base(num_classes=num_classes, drop_path_rate=dropout_rate),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name]()


# ===================== Scheduler Factory =====================

def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    params: Dict[str, Any],
    num_epochs: int,
    steps_per_epoch: int,
    num_warmup_steps: int = 500
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler."""
    
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = min(num_warmup_steps, total_steps // 4)
    decay_steps = total_steps - warmup_steps
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    if scheduler_name == 'cosine':
        eta_min = params['learning_rate'] * 0.01
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=eta_min
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_name == 'step':
        step_size = params.get('step_size', 10) * steps_per_epoch
        gamma = params.get('step_gamma', 0.1)
        decay_scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
    
    elif scheduler_name == 'onecycle':
        max_lr = params['learning_rate'] * params.get('max_lr_factor', 10.0)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
    
    elif scheduler_name == 'linear':
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=decay_steps
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_steps]
        )
    
    return None


# ===================== Objective Function =====================

class HPOObjective:
    """Objective function for Optuna optimization."""
    
    def __init__(
        self,
        model_name: str,
        config: HPOConfig,
        dataloaders: Dict,
        dataset_sizes: Dict,
        num_classes: int,
        device: torch.device,
        finetune_from: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.config = config
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.num_classes = num_classes
        self.device = device
        self.finetune_from = finetune_from
    
    def __call__(self, trial: 'Trial') -> float:
        """Run a single trial and return the validation accuracy."""
        
        # Set seed for reproducibility within trial
        seed = self.config.seed + trial.number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Get hyperparameters for this trial
        params = get_search_space(trial, self.finetune_from)
        
        # Create model
        model = create_model(
            self.model_name,
            self.num_classes,
            dropout_rate=params['dropout_rate']
        )
        
        # Create optimizer
        if params['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        
        # Create scheduler
        steps_per_epoch = len(self.dataloaders['train'])
        scheduler = create_scheduler(
            optimizer,
            params['scheduler'],
            params,
            self.config.n_epochs_per_trial,
            steps_per_epoch,
            self.config.n_warmup_steps
        )
        
        # Create criterion with label smoothing
        criterion = LabelSmoothingCrossEntropy(
            smoothing=params['label_smoothing']
        )
        
        # Create trainer with Optuna trial for pruning
        trainer = Trainer(
            model,
            dataloaders=self.dataloaders,
            dataset_sizes=self.dataset_sizes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            num_epochs=self.config.n_epochs_per_trial,
            save_path=None,  # Don't save during HPO
            optuna_trial=trial,
            optuna_prune_metric='val_acc'
        )
        
        try:
            # Run training
            model, history = trainer.fasttrain()  # Use fast training with AMP
            
            # Return best validation accuracy
            return float(trainer.best_acc)
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0  # Return worst possible value


# ===================== HPO Runner =====================

class HPORunner:
    """Main HPO runner class."""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup storage
        self.storage_url = f"sqlite:///{config.storage_path}"
    
    def load_data(self) -> Tuple[Dict, Dict, List[str], int]:
        """Load the dataset."""
        dataset_name = self.config.dataset
        
        # Use 224x224 version of CIFAR100 for models that need larger inputs
        if dataset_name == 'cifar100':
            dataset_name = 'cifar100_224'
        
        if dataset_name in ['mnist', 'fashionmnist', 'cifar100', 'cifar100_224', 'caltech101']:
            return prepare_builtin_data(
                data_dir=f"data/{self.config.dataset}",
                batch_size=self.config.batch_size,
                dataset=dataset_name
            )
        elif self.config.dataset == 'intel':
            train_dir = 'data/intel_image/seg_train/seg_train'
            test_dir = 'data/intel_image/seg_test/seg_test'
            return prepare_data(
                train_dir=train_dir,
                test_dir=test_dir,
                input_size=self.config.input_size,
                batch_size=self.config.batch_size,
                dataset=self.config.dataset
            )
        elif self.config.dataset == 'mit':
            train_dir = 'data/mit_indoor/indoorCVPR_09/Images'
            test_dir = 'data/mit_indoor/TestImages.txt'
            return prepare_data(
                train_dir=train_dir,
                test_dir=test_dir,
                input_size=self.config.input_size,
                batch_size=self.config.batch_size,
                dataset=self.config.dataset
            )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
    
    def create_study(
        self,
        study_name: str,
        direction: str = 'maximize'
    ) -> 'optuna.Study':
        """Create or load an Optuna study."""
        
        # Create pruner
        pruner = MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps_pruner,
            interval_steps=1
        )
        
        # Create sampler with seed for reproducibility
        sampler = TPESampler(seed=self.config.seed)
        
        # Create or load study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction=direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True
        )
        
        return study
    
    def run_group_hpo(
        self,
        group_name: str,
        representative_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run HPO for a model group.
        
        Args:
            group_name: Name of the model group ('legacy', 'resnet', 'modern')
            representative_model: Optional specific model to use for HPO
            
        Returns:
            Best hyperparameters found
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not installed. Run 'pip install optuna'")
        
        if group_name not in MODEL_GROUPS:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(MODEL_GROUPS.keys())}")
        
        # Use first model in group as representative if not specified
        if representative_model is None:
            representative_model = MODEL_GROUPS[group_name][0]
        
        print(f"\n{'='*60}")
        print(f"Running Group HPO: {group_name}")
        print(f"Representative Model: {representative_model}")
        print(f"Number of Trials: {self.config.n_trials}")
        print(f"{'='*60}\n")
        
        # Load data
        dataloaders, dataset_sizes, class_names, num_classes = self.load_data()
        print(f"Dataset: {self.config.dataset}, Classes: {num_classes}")
        
        # Create study
        study_name = f"{self.config.study_name}_{group_name}"
        study = self.create_study(study_name)
        
        # Create objective
        objective = HPOObjective(
            model_name=representative_model,
            config=self.config,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_classes=num_classes,
            device=self.device
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n{'='*60}")
        print(f"Group HPO Complete: {group_name}")
        print(f"Best Validation Accuracy: {best_value:.4f}")
        print(f"Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
        
        # Save best params
        self._save_group_config(group_name, best_params, best_value)
        
        return best_params
    
    def run_model_finetune(
        self,
        model_name: str,
        group_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run fine-tuning HPO for a specific model.
        
        Args:
            model_name: Name of the model
            group_params: Best parameters from group HPO
            
        Returns:
            Fine-tuned hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is not installed. Run 'pip install optuna'")
        
        print(f"\n{'='*60}")
        print(f"Running Fine-Tune HPO: {model_name}")
        print(f"Number of Trials: {self.config.n_finetune_trials}")
        print(f"Starting from group params...")
        print(f"{'='*60}\n")
        
        # Load data
        dataloaders, dataset_sizes, class_names, num_classes = self.load_data()
        
        # Create study
        study_name = f"{self.config.study_name}_{model_name}_finetune"
        study = self.create_study(study_name)
        
        # Create objective with fine-tuning
        objective = HPOObjective(
            model_name=model_name,
            config=self.config,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_classes=num_classes,
            device=self.device,
            finetune_from=group_params
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_finetune_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\n{'='*60}")
        print(f"Fine-Tune HPO Complete: {model_name}")
        print(f"Best Validation Accuracy: {best_value:.4f}")
        print(f"Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}\n")
        
        # Save model-specific config
        self._save_model_config(model_name, best_params, best_value)
        
        return best_params
    
    def run_full_hpo(self):
        """Run full HPO pipeline: group HPO + per-model fine-tuning."""
        
        print("\n" + "="*80)
        print("STARTING FULL HPO PIPELINE")
        print("="*80 + "\n")
        
        all_results = {}
        
        # Phase 1: Group HPO
        print("\n[PHASE 1] Running Group HPO...\n")
        group_best_params = {}
        
        for group_name in MODEL_GROUPS.keys():
            best_params = self.run_group_hpo(group_name)
            group_best_params[group_name] = best_params
            all_results[f'group_{group_name}'] = best_params
        
        # Phase 2: Per-Model Fine-Tuning
        print("\n[PHASE 2] Running Per-Model Fine-Tuning...\n")
        
        for group_name, models in MODEL_GROUPS.items():
            group_params = group_best_params[group_name]
            
            for model_name in models:
                best_params = self.run_model_finetune(model_name, group_params)
                all_results[f'model_{model_name}'] = best_params
        
        # Save summary
        self._save_summary(all_results)
        
        print("\n" + "="*80)
        print("FULL HPO PIPELINE COMPLETE")
        print(f"Results saved to: {self.config.output_dir}")
        print("="*80 + "\n")
        
        return all_results
    
    def _save_group_config(
        self,
        group_name: str,
        params: Dict[str, Any],
        best_value: float
    ):
        """Save group configuration to JSON."""
        config = {
            'hpo_info': {
                'group': group_name,
                'best_val_acc': best_value,
                'n_trials': self.config.n_trials,
                'dataset': self.config.dataset,
                'timestamp': datetime.now().isoformat()
            },
            'dataset_info': {
                'dataset': self.config.dataset,
                'input_size': self.config.input_size,
                'batch_size': self.config.batch_size
            },
            'train_info': {
                'optimizer': params['optimizer'],
                'learning_rate': params['learning_rate'],
                'weight_decay': params.get('weight_decay', 1e-4),
                'scheduler': params['scheduler'],
                'dropout_rate': params['dropout_rate'],
                'label_smoothing': params['label_smoothing']
            }
        }
        
        # Add scheduler-specific params
        if params['scheduler'] == 'step':
            config['train_info']['step_size'] = params.get('step_size', 10)
            config['train_info']['step_gamma'] = params.get('step_gamma', 0.1)
        elif params['scheduler'] == 'onecycle':
            config['train_info']['max_lr_factor'] = params.get('max_lr_factor', 10.0)
        
        filepath = os.path.join(self.config.output_dir, f'group_{group_name}_best.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved group config to: {filepath}")
    
    def _save_model_config(
        self,
        model_name: str,
        params: Dict[str, Any],
        best_value: float
    ):
        """Save model-specific configuration to JSON."""
        config = {
            'hpo_info': {
                'model': model_name,
                'best_val_acc': best_value,
                'n_trials': self.config.n_finetune_trials,
                'dataset': self.config.dataset,
                'timestamp': datetime.now().isoformat()
            },
            'dataset_info': {
                'dataset': self.config.dataset,
                'input_size': self.config.input_size,
                'batch_size': self.config.batch_size
            },
            'model_info': {
                'name': model_name,
                'dropout_rate': params['dropout_rate']
            },
            'train_info': {
                'optimizer': params['optimizer'],
                'learning_rate': params['learning_rate'],
                'weight_decay': params.get('weight_decay', 1e-4),
                'scheduler': params['scheduler'],
                'label_smoothing': params['label_smoothing']
            }
        }
        
        # Add scheduler-specific params
        if params['scheduler'] == 'step':
            config['train_info']['step_size'] = params.get('step_size', 10)
            config['train_info']['step_gamma'] = params.get('step_gamma', 0.1)
        elif params['scheduler'] == 'onecycle':
            config['train_info']['max_lr_factor'] = params.get('max_lr_factor', 10.0)
        
        filepath = os.path.join(self.config.output_dir, f'{model_name}_hpo_optimized.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved model config to: {filepath}")
    
    def _save_summary(self, all_results: Dict[str, Any]):
        """Save HPO summary."""
        summary = {
            'config': asdict(self.config),
            'results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.config.output_dir, 'hpo_summary.json')
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved HPO summary to: {filepath}")


# ===================== CLI =====================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Optimization with Optuna',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'group', 'finetune', 'single'],
                        help='HPO mode: full pipeline, group only, model fine-tune, or single model')
    
    # Group/Model selection
    parser.add_argument('--group', type=str, default=None,
                        choices=['legacy', 'resnet', 'modern'],
                        help='Model group for group HPO mode')
    parser.add_argument('--model', type=str, default=None,
                        choices=ALL_MODELS,
                        help='Model name for finetune/single mode')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'mnist', 'fashionmnist', 'intel', 'mit', 'caltech101'],
                        help='Dataset to use for HPO')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # HPO settings
    parser.add_argument('--trials', type=int, default=70,
                        help='Number of HPO trials for group optimization')
    parser.add_argument('--finetune_trials', type=int, default=15,
                        help='Number of fine-tuning trials per model')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs per trial')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps for scheduler')
    
    # Optuna settings
    parser.add_argument('--study_name', type=str, default='hpo_study',
                        help='Base name for Optuna studies')
    parser.add_argument('--storage', type=str, default='hpo_results.db',
                        help='SQLite database path for Optuna storage')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel trials (1 for sequential, -1 for auto)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='config/hpo',
                        help='Output directory for optimized configs')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is not installed. Run 'pip install optuna' to use HPO.")
        sys.exit(1)
    
    # Create config
    config = HPOConfig(
        dataset=args.dataset,
        input_size=args.input_size,
        batch_size=args.batch_size,
        n_trials=args.trials,
        n_finetune_trials=args.finetune_trials,
        n_epochs_per_trial=args.epochs,
        n_warmup_steps=args.warmup_steps,
        study_name=args.study_name,
        storage_path=args.storage,
        n_jobs=args.n_jobs,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Create runner
    runner = HPORunner(config)
    
    # Run based on mode
    if args.mode == 'full':
        runner.run_full_hpo()
    
    elif args.mode == 'group':
        if args.group is None:
            print("Error: --group is required for group mode")
            sys.exit(1)
        runner.run_group_hpo(args.group)
    
    elif args.mode == 'finetune':
        if args.model is None:
            print("Error: --model is required for finetune mode")
            sys.exit(1)
        
        # Find which group this model belongs to
        group_name = None
        for gname, models in MODEL_GROUPS.items():
            if args.model in models:
                group_name = gname
                break
        
        if group_name is None:
            print(f"Error: Model {args.model} not found in any group")
            sys.exit(1)
        
        # Load group params
        group_config_path = os.path.join(config.output_dir, f'group_{group_name}_best.json')
        if not os.path.exists(group_config_path):
            print(f"Error: Group config not found: {group_config_path}")
            print("Run group HPO first: python hpo.py --mode group --group {group_name}")
            sys.exit(1)
        
        with open(group_config_path, 'r') as f:
            group_config = json.load(f)
        
        group_params = group_config['train_info']
        runner.run_model_finetune(args.model, group_params)
    
    elif args.mode == 'single':
        if args.model is None:
            print("Error: --model is required for single mode")
            sys.exit(1)
        
        # Run single model HPO (no group params)
        print(f"\nRunning single model HPO for: {args.model}")
        dataloaders, dataset_sizes, class_names, num_classes = runner.load_data()
        
        study_name = f"{config.study_name}_{args.model}_single"
        study = runner.create_study(study_name)
        
        objective = HPOObjective(
            model_name=args.model,
            config=config,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_classes=num_classes,
            device=runner.device
        )
        
        study.optimize(
            objective,
            n_trials=config.n_trials,
            n_jobs=config.n_jobs,
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        runner._save_model_config(args.model, study.best_params, study.best_value)


if __name__ == '__main__':
    main()
