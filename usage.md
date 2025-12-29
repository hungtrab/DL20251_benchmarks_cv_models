# Project Usage Guide

This document describes how to run the scripts in this repository and recommended run order for a full benchmarking workflow (HPO → training → evaluation → statistical comparison).

---

**Quick overview (recommended run order)**

1. Environment setup
2. Prepare datasets (resize/split/augment)
3. Run Hyperparameter Optimization (`hpo.py`)
4. Train / fine-tune with best configurations (`train.py`, `finetune.py`)
5. Run comprehensive evaluation (`evaluate.py`) — robustness, calibration, bootstrap CI
6. Pairwise statistical comparison (`pairwise_model_comparison` in `evaluate.py`)
7. Collect artifacts: `logs/`, `results/`, `config/hpo/*.json`, `hpo_results.db`

---

## 1. Environment

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- GPU: if using CUDA, ensure the environment's PyTorch build matches your CUDA driver.
- If you add packages (e.g., a different scheduler or visualization tool), update `requirements.txt`.

---

## 2. Data preparation

Use `data_preprocess.py` or the example helper `dataset_usage_examples.py` to prepare datasets and ensure correct input sizes.

Example (resize dataset to 224×224 for ImageNet-style models):

```bash
python data_preprocess.py --dataset cifar100 --out data/cifar100_224 --size 224
# or use the provided helper
python dataset_usage_examples.py --prepare cifar100 --out data/cifar100_224
```

Output structure should be consistent with other scripts (train/evaluate): organized train/val/test folders.

---

## 3. Hyperparameter optimization (HPO)

`hpo.py` runs Optuna experiments. Recommended approach: run group-level HPO (70 trials) to find per-group good defaults, then per-model fine-tuning if desired.

Basic group run (SQLite persistence, MedianPruner):

```bash
python hpo.py --mode group --trials 70 --n_jobs 4 --storage sqlite:///hpo_results.db
```

Single-model tuning:

```bash
python hpo.py --mode single --model resnet50 --trials 70 --storage sqlite:///hpo_results.db
```

After HPO finishes, optimized configs are stored in `config/hpo/` (e.g., `config/hpo/resnet50_hpo_optimized.json`). Use these for training.

**Inspecting HPO results:**

Use `inspect_hpo.py` to analyze trial results from the database:

```bash
# List all studies in the database
python inspect_hpo.py --storage hpo_results.db --list

# Show top 10 trials for a specific study
python inspect_hpo.py --storage hpo_results.db --study hpo_study_legacy --top 10

# Export all trials to CSV for analysis
python inspect_hpo.py --storage hpo_results.db --study hpo_study_legacy --export

# Compare all studies
python inspect_hpo.py --storage hpo_results.db --compare
```

**What's logged in the database:**
- **Trial parameters**: learning rate, weight decay, dropout, optimizer, scheduler, label smoothing
- **Trial value**: validation accuracy (maximize)
- **Trial state**: COMPLETE, PRUNED, or FAILED
- **Intermediate values**: validation accuracy at each epoch (for pruning)
- **Timestamps**: start and end times for each trial

**How to identify good trials:**
- Trials are ranked by validation accuracy (higher is better)
- Use `inspect_hpo.py` to see the top trials and their hyperparameters
- The best trial's parameters are automatically saved to `config/hpo/<model>_hpo_optimized.json`
- Compare across studies using `--compare` to see which model group performs best

Tips:
- Use `--n_jobs` to parallelize multiple trials if you have multiple GPUs or a cluster.
- The HPO module uses a `MedianPruner` and `TPESampler` by default; change via `hpo.py` if needed.
- Database persists all trials — you can resume HPO or add more trials later.

---

## 4. Training and fine-tuning

Primary training entry points:
- `train.py`: full training loop (supports Mixup/CutMix, SAM, label smoothing, adaptive training via `trainer.py` config)
- `pretrain.py`: pretraining flows if used
- `finetune.py`: fine-tune a checkpoint on a new dataset
- `trainer.py`: core training utilities (SAM, mixup/cutmix functions, adaptive logic)

Train using a config file (from HPO or default configs in `config/`):

```bash
python train.py --config config/resnet50_adamw.json --num_epochs 100 --batch_size 32
```

**Basic training options:**

```bash
# Basic training with default settings
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100

# Training with validation split (20% of training data)
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 --val --val_size 0.2

# Training with custom optimizer and scheduler
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 \
    --optimizer adamw --learning_rate 0.001 \
    --scheduler cosine --num_warmup_steps 500

# Training with step scheduler (step every 10 epochs, gamma=0.1)
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 \
    --scheduler step --step_size 10 --step_gamma 0.1

# Training with onecycle scheduler (max_lr = lr * 10)
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 \
    --scheduler onecycle --max_lr_factor 10.0

# Fast training with mixed precision (AMP)
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 --fasttrain

# Resume from checkpoint
python train.py --model_name resnet50 --dataset cifar100 --num_epochs 100 \
    --resume results/best_model.pth
```

**ConvNeXt V2 with self-supervised pretraining (FCMAE):**

```bash
# ConvNeXt V2 with FCMAE pretraining (50 epochs) followed by supervised training
python train.py --model_name convnextv2_t --dataset cifar100 --num_epochs 100 \
    --conv_pretrain --conv_pretrain_epochs 50 \
    --conv_mask_ratio 0.6 --conv_pretrain_lr 1.5e-4 --conv_pretrain_wd 0.05

# ConvNeXt V2 Base with extended pretraining
python train.py --model_name convnextv2_b --dataset intel --num_epochs 200 \
    --conv_pretrain --conv_pretrain_epochs 100 \
    --conv_mask_ratio 0.7 --val --val_size 0.2

# Use pretrained checkpoint (without FCMAE)
python train.py --model_name convnextv2_t --dataset cifar100 --num_epochs 100 \
    --pretrained_path weights/convnext_pretrained.pth
```

**Automatic post-training evaluation (Section 6):**

The `train.py` script now supports automatic comprehensive evaluation after training completes:

```bash
# Basic training (only top-1/5 accuracy, confusion matrix)
python train.py --config config/resnet50_adamw.json --num_epochs 100

# Training + Section 4.3 features (Mixup, CutMix, SAM, Adaptive Training)
python train.py --config config/resnet50_adamw.json --num_epochs 100 \
    --use_mixup --mixup_alpha 0.2 \
    --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 \
    --use_sam --sam_rho 0.05 \
    --adaptive_training --adaptive_check_interval 5

# Training + robustness evaluation (Gaussian noise, salt & pepper, blur)
python train.py --config config/resnet50_adamw.json --num_epochs 100 --eval_robustness

# Training + calibration evaluation (ECE, reliability diagram)
python train.py --config config/resnet50_adamw.json --num_epochs 100 --eval_calibration

# Training + bootstrap CI (1000 samples)
python train.py --config config/resnet50_adamw.json --num_epochs 100 --eval_bootstrap

# Training + efficiency benchmark (throughput, latency, VRAM, model size)
python train.py --config config/resnet50_adamw.json --num_epochs 100 --eval_efficiency

# FULL benchmark (Section 4.3 + Section 6 + Section 2 - ALL metrics)
python train.py --config config/resnet50_adamw.json --num_epochs 100 \
    --use_mixup --use_cutmix --use_sam --adaptive_training \
    --eval_full --eval_efficiency

# Customize bootstrap samples and calibration bins
python train.py --config config/resnet50_adamw.json --num_epochs 100 \
    --eval_full --n_bootstrap 2000 --n_calibration_bins 20
```

**Complete CLI Arguments Reference:**

**Dataset & Data Options:**
- `--dataset`: Dataset name (mnist, fashionmnist, cifar100, caltech101, intel, mit, imagenet)
- `--input_size`: Input image size (default: 224)
- `--batch_size`: Batch size for training/validation (default: 32)
- `--val`: Enable validation set (split from training data)
- `--val_size`: Validation set size as fraction of training data (default: 0.2)

**Model Options:**
- `--model_name`: Model architecture (alexnet, lenet, vgg16, vgg16_bn, resnet18/34/50/101, mobilenetv3_s/l, efficientnetv2_s/m/l, vit, convnextv2_t/b)
- `--dropout_rate`: Dropout rate (default: 0.4)
- `--pretrained_path`: Path to pretrained checkpoint

**Training Hyperparameters:**
- `--num_epochs`: Number of training epochs (default: 25)
- `--learning_rate`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer (adam, adamw, sgd)
- `--criterion`: Loss function (cross_entropy, mse, hinge)
- `--scheduler`: Learning rate scheduler (constant, linear, cosine, step, onecycle)
- `--num_warmup_steps`: Number of warmup steps (default: 500)

**Learning Rate Schedulers:**
- `--scheduler constant`: No scheduler (constant learning rate)
- `--scheduler linear`: Linear decay with warmup
- `--scheduler cosine`: Cosine annealing with warmup
- `--scheduler step`: Step decay with warmup
  - `--step_size`: Step size in epochs (default: 10)
  - `--step_gamma`: Gamma multiplier (default: 0.1)
- `--scheduler onecycle`: OneCycleLR (handles warmup internally)
  - `--max_lr_factor`: Max LR multiplier (default: 10.0)

**Section 4.3 - Anti-Overfitting & Adaptive Training:**
- `--use_mixup`: Enable Mixup data augmentation
- `--mixup_alpha`: Mixup alpha parameter (default: 0.2)
- `--use_cutmix`: Enable CutMix data augmentation
- `--cutmix_alpha`: CutMix alpha parameter (default: 1.0)
- `--cutmix_prob`: Probability of applying CutMix (default: 0.5)
- `--use_sam`: Enable SAM (Sharpness-Aware Minimization) optimizer
- `--sam_rho`: SAM rho parameter for perturbation (default: 0.05)
- `--label_smoothing`: Label smoothing factor (default: 0.0)
- `--adaptive_training`: Enable adaptive training (auto-adjust WD, aug, SAM)
- `--adaptive_check_interval`: Check interval for adaptive training in epochs (default: 5)

**ConvNeXt V2 Self-Supervised Pretraining (FCMAE):**
- `--conv_pretrain`: Enable FCMAE pretraining for ConvNeXt V2 models
- `--conv_pretrain_epochs`: Number of pretraining epochs (default: 50)
- `--conv_mask_ratio`: Masking ratio for FCMAE (default: 0.6)
- `--conv_pretrain_lr`: Pretraining learning rate (default: 1.5e-4)
- `--conv_pretrain_wd`: Pretraining weight decay (default: 0.05)

**Section 6 - Comprehensive Evaluation:**
- `--eval_robustness`: Robustness testing with noise injection (Section 6.1)
- `--eval_calibration`: ECE and reliability diagram (Section 6.2)
- `--eval_bootstrap`: Bootstrap confidence interval (Section 6.3)
- `--eval_full`: Enable all comprehensive metrics above
- `--n_bootstrap`: Number of bootstrap samples (default: 1000)
- `--n_calibration_bins`: Number of calibration bins (default: 15)

**Section 2 - Efficiency Benchmark:**
- `--eval_efficiency`: Benchmark efficiency (throughput, latency, VRAM, model size)

**Logging & Monitoring:**
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name (default: 'dl20251-cv')
- `--wandb_entity`: W&B entity/team name
- `--wandb_run_name`: Custom W&B run name
- `--use_tensorboard`: Enable TensorBoard logging
- `--tensorboard_log_dir`: TensorBoard log directory (default: 'logs')

**Other Options:**
- `--fasttrain`: Enable mixed precision training (torch.cuda.amp)
- `--resume`: Path to checkpoint to resume training from
- `--seed`: Random seed for reproducibility (default: 42)
- `--use_class_weights`: Use class weights for imbalanced datasets
- `--weight_type`: Type of class weights ('inverse' or 'sqrt_inverse')
- `--config`: Path to JSON config file (overrides defaults)

All evaluation results are saved to the experiment directory and automatically logged to W&B/TensorBoard if enabled.

**Weight Initialization:**
- All models now use **He initialization** (kaiming_normal_) for Conv2d and Linear layers
- BatchNorm layers initialized with weight=1, bias=0
- Improves convergence for ReLU-based architectures

Fine-tune a checkpoint:

```bash
python finetune.py --checkpoint logs/resnet50_exp/best.pth --config config/resnet50_finetune.json --epochs 30
```

Important training options (config keys or CLI flags exposed by your scripts):
- Mixup: `use_mixup` (bool), `mixup_alpha` (float)
- CutMix: `use_cutmix` (bool), `cutmix_alpha` (float), `cutmix_prob` (float)
- SAM (Sharpness-Aware Minimization): `use_sam` (bool), `sam_rho` (float)
- Label smoothing: `label_smoothing` (float)
- Adaptive training: enable and configure `AdaptiveTrainingConfig` in `trainer.py` or config JSON

Use HPO-produced `config/hpo/*_optimized.json` for the `--config` argument to train with tuned hyperparameters.

---

## 5. Quick/demo runs

`demo.py` and `app.py` provide quick inference or web/demo interfaces. Use them for sanity checks or visual inspection of predictions.

Example:

```bash
python demo.py --model logs/resnet50_exp/best.pth --input samples/example.jpg
```

---

## 6. Evaluation (robustness, calibration, bootstrap)

`evaluate.py` provides comprehensive evaluation utilities. Key functions/flags:
- `evaluate_model(...)` — base evaluation: top-1/top-5, confusion matrix, classification report
- `full_benchmark_evaluation(...)` or CLI `--full` — runs robustness tests, calibration (ECE), bootstrap CI
- `pairwise_model_comparison(...)` — runs pairwise McNemar's tests across models

CLI usage example (full evaluation):

```bash
python evaluate.py --model logs/resnet50_exp/best.pth --dataset cifar100 --batch_size 64 --save_path results/resnet50 --full
```

Available CLI flags (evaluate.py):
- `--robustness` : compute robustness metrics (Gaussian noise, salt & pepper, blur)
- `--calibration` : compute ECE and save reliability diagram
- `--bootstrap` : compute bootstrap confidence interval for accuracy
- `--full` : enable all of the above

Programmatic usage (Python):

```python
from evaluate import full_benchmark_evaluation
results = full_benchmark_evaluation(model, test_loader, num_class=100, save_path='results/resnet50')
```

Outputs:
- Confusion matrix: `results/confusion_matrix.png`
- Reliability diagram: `results/reliability_diagram.png`
- Numerical results returned as a dictionary and stored in `results/`

---

## 7. Pairwise statistical comparisons

To compare multiple models and test statistical differences, use `pairwise_model_comparison` in `evaluate.py`:

```python
from evaluate import pairwise_model_comparison
models = {
  'resnet50': load_model('logs/resnet50_exp/best.pth'),
  'convnext': load_model('logs/convnext_exp/best.pth')
}
# test_loader: DataLoader for test set
df = pairwise_model_comparison(models, test_loader, save_path='results/')
```

This runs McNemar's test and writes `results/pairwise_comparison.csv` with p-values, chi-squared, significance flags, and effect sizes.

---

## 8. Scripts reference (short)

- `app.py`: web/demo interface (if provided)
- `checkwd.py`: quick working-dir checks / sanity scripts
- `data_preprocess.py`: dataset resizing / normalizing / splitting utilities
- `dataset_usage_examples.py`: example dataset download & preprocessing helpers
- `demo.py`: quick single-image inference demo
- `evaluate.py`: model evaluation and statistical tests (robustness, ECE, bootstrap, McNemar)
- `finetune.py`: fine-tuning flow
- `hpo.py`: Optuna HPO workflows
- `inspect_hpo.py`: Inspect and analyze Optuna trial results from database
- `model.py`, `models_dense.py`, `codeae/model.py`: model definitions
- `pretrain.py`: pretraining flows (if provided)
- `train.py`: full training pipeline with integrated Section 6 evaluation
- `trainer.py`: training primitives (SAM, mixup/cutmix, label smoothing, adaptive logic)

**Quick Reference: Evaluation Metrics**

| Flag | Section | Metrics Computed | Output Files |
|------|---------|------------------|--------------|
| *(default)* | Basic | Top-1/5 accuracy, confusion matrix | `confusion_matrix.png` |
| `--eval_robustness` | 6.1 | Gaussian noise, salt & pepper, blur robustness | *(logged to metrics)* |
| `--eval_calibration` | 6.2 | ECE, reliability diagram | `reliability_diagram.png` |
| `--eval_bootstrap` | 6.3 | Bootstrap CI (95%) | *(logged to metrics)* |
| `--eval_full` | 6.1-6.3 | All of the above | All outputs |

---

## 9. Reproducibility & artifacts

- Store HPO DB: `hpo_results.db` (or other SQLite path you passed to `hpo.py`)
- Save optimized configs: `config/hpo/*.json`
- Save logs & checkpoints: `logs/<exp_name>/` (contains checkpoints, tensorboard, etc.)
- Save evaluation outputs: `results/<exp_name>/`
- Commit code and configs alongside final weights for reproducibility.

---

## 10. Troubleshooting tips

- OOM on GPU: reduce `--batch_size` or use mixed precision training (AMP) if available.
- Slow HPO: reduce the number of epochs in the trial objective or use a pruner (`MedianPruner` recommended).
- Evaluation errors: ensure `test_dataloader` uses the same normalization and input size as training.
- Mismatched classes: ensure `num_class` argument matches dataset labels.

---

## 11. Example end-to-end run (summary)

```bash
# 1. Setup env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data
python data_preprocess.py --dataset cifar100 --out data/cifar100_224 --size 224

# 3. HPO (group-level)
python hpo.py --mode group --trials 70 --n_jobs 4 --storage hpo_results.db

# 4. Train with selected config + FULL comprehensive evaluation
python train.py --config config/hpo/resnet50_hpo_optimized.json \
    --num_epochs 100 --batch_size 32 \
    --val --val_size 0.2 \
    --eval_full --eval_efficiency \
    --use_tensorboard --use_wandb

# 4b. Train ConvNeXt V2 with FCMAE pretraining + validation split
python train.py --model_name convnextv2_t --dataset cifar100 \
    --conv_pretrain --conv_pretrain_epochs 50 \
    --num_epochs 100 --batch_size 32 \
    --val --val_size 0.2 \
    --use_mixup --use_cutmix --use_sam \
    --eval_full --eval_efficiency \
    --use_wandb

# 5. (Optional) Standalone evaluation if needed later
python evaluate.py --model logs/resnet50_exp/best.pth \
    --dataset cifar100 --batch_size 64 \
    --save_path results/resnet50 --full

# 6. Inspect HPO results
python inspect_hpo.py --storage hpo_results.db --top 10

# 7. Pairwise comparisons (python snippet)
python - <<'PY'
from evaluate import pairwise_model_comparison
from dataset_usage_examples import get_test_loader
from some_util import load_model_checkpoint

test_loader = get_test_loader('cifar100', batch_size=64)
models = {
  'resnet50': load_model_checkpoint('logs/resnet50_exp/best.pth'),
  'convnext': load_model_checkpoint('logs/convnext_exp/best.pth')
}

pairwise_model_comparison(models, test_loader, save_path='results/')
PY
```

**Note:** With `--eval_full` flag in step 4, the training script automatically runs:
- Robustness testing (Gaussian noise, salt & pepper, blur)
- Calibration evaluation (ECE, reliability diagram)
- Bootstrap confidence interval (1000 samples)

All metrics are saved to the experiment directory and logged to W&B/TensorBoard.

---

## 12. Complete Training Examples

**Example 1: Basic ResNet50 training with validation split**
```bash
python train.py --model_name resnet50 --dataset cifar100 \
    --num_epochs 100 --batch_size 64 \
    --optimizer adamw --learning_rate 0.001 \
    --scheduler cosine --num_warmup_steps 500 \
    --val --val_size 0.2 \
    --use_tensorboard
```

**Example 2: ConvNeXt V2 with FCMAE pretraining + full evaluation**
```bash
python train.py --model_name convnextv2_t --dataset intel \
    --conv_pretrain --conv_pretrain_epochs 50 \
    --conv_mask_ratio 0.6 \
    --num_epochs 100 --batch_size 32 \
    --optimizer adamw --learning_rate 0.001 \
    --scheduler cosine \
    --val --val_size 0.2 \
    --use_mixup --mixup_alpha 0.2 \
    --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 \
    --use_sam --sam_rho 0.05 \
    --adaptive_training \
    --eval_full --eval_efficiency \
    --use_wandb --use_tensorboard
```

**Example 3: VGG16 with step scheduler and class weights**
```bash
python train.py --model_name vgg16_bn --dataset caltech101 \
    --num_epochs 150 --batch_size 32 \
    --optimizer sgd --learning_rate 0.01 \
    --scheduler step --step_size 30 --step_gamma 0.1 \
    --val --val_size 0.15 \
    --use_class_weights --weight_type inverse \
    --dropout_rate 0.5 \
    --fasttrain
```

**Example 4: EfficientNetV2 with OneCycle scheduler + Mixup/CutMix**
```bash
python train.py --model_name efficientnetv2_s --dataset cifar100 \
    --num_epochs 200 --batch_size 64 \
    --optimizer adamw --learning_rate 0.0005 \
    --scheduler onecycle --max_lr_factor 15.0 \
    --val --val_size 0.2 \
    --use_mixup --mixup_alpha 0.3 \
    --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 \
    --label_smoothing 0.1 \
    --eval_full --eval_efficiency \
    --fasttrain --use_wandb
```

**Example 5: Resume training from checkpoint**
```bash
python train.py --model_name resnet50 --dataset cifar100 \
    --resume results/demo1_cifar100_resnet50_adamw_20251229/checkpoint_final.pth \
    --num_epochs 150 --batch_size 64 \
    --val --val_size 0.2 \
    --eval_full
```

**Example 6: HPO-optimized config with all features**
```bash
python train.py --config config/hpo/resnet50_hpo_optimized.json \
    --num_epochs 100 --batch_size 32 \
    --val --val_size 0.2 \
    --use_mixup --use_cutmix --use_sam \
    --adaptive_training --adaptive_check_interval 5 \
    --eval_full --eval_efficiency \
    --n_bootstrap 2000 --n_calibration_bins 20 \
    --fasttrain --use_wandb --use_tensorboard
```

---

## 13. Next steps & customization

- Adjust HPO search space in `hpo.py` for new parameters.
- If you add new data augmentations, expose flags in `trainer.py` and update config schema.
- Automate runs on multiple GPUs / cluster using Slurm or similar; ensure `--n_jobs` and storage path for Optuna are tuned for distributed runs.

---

If you'd like, I can now:
- Add a one-line `README.md` linking to `usage.md` and the `plan.md` roadmap, or
- Update scripts to add CLI flags (if any are still missing) so all behaviors are CLI-configurable.

