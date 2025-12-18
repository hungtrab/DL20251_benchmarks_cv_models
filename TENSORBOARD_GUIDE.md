# TensorBoard Logging Guide

This guide explains how to use TensorBoard logging in the training pipeline.

## Features

TensorBoard integration provides:
- **Real-time metrics**: Loss and accuracy for train/test phases
- **Learning rate tracking**: Monitor LR changes during training
- **Hyperparameter logging**: Compare different configurations
- **Configuration backup**: Auto-save training config to logs folder

## Usage

### Basic Training with TensorBoard

Enable TensorBoard logging with the `--use_tensorboard` flag:

```bash
python train.py --config config/resnet18_adam.json --use_tensorboard
```

### Custom Log Directory

Specify a custom logs directory (default is `logs`):

```bash
python train.py --config config/mobilenetv3_adam.json --use_tensorboard --tensorboard_log_dir my_logs
```

### Combined with W&B

You can use both TensorBoard and W&B simultaneously:

```bash
python train.py --config config/vgg16_adam.json --use_tensorboard --use_wandb
```

## Viewing TensorBoard Results

After training starts, launch TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open your browser to: **http://localhost:6006**

### View Specific Experiment

To view only a specific experiment:

```bash
tensorboard --logdir=logs/intel_resnet18_20251218_143025
```

### Compare Multiple Runs

TensorBoard automatically compares all runs in the log directory:

```bash
tensorboard --logdir=logs
```

## What Gets Logged

### Per-Epoch Metrics
- `train/loss`: Training loss
- `train/acc`: Training accuracy
- `test/loss`: Validation loss
- `test/acc`: Validation accuracy
- `lr`: Current learning rate (train phase only)

### Final Summary
- `best_val_acc`: Best validation accuracy achieved
- `best_val_loss`: Best validation loss achieved
- Hyperparameters: lr, batch_size, optimizer, model, dataset

## Directory Structure

After running with TensorBoard enabled:

```
logs/
├── intel_resnet18_20251218_143025/
│   ├── events.out.tfevents.xxx    # TensorBoard log file
│   └── config.json                 # Training configuration
├── mnist_lenet_20251218_150130/
│   ├── events.out.tfevents.xxx
│   └── config.json
└── ...
```

## Configuration Backup

Your training configuration is automatically saved to:
```
logs/<experiment_name>/config.json
```

This includes all arguments (from config file + CLI overrides), making it easy to reproduce experiments.

## Example Commands

### Train ResNet18 on Intel dataset with TensorBoard
```bash
python train.py \
  --config config/resnet18_adam.json \
  --use_tensorboard \
  --num_epochs 50
```

### Train MobileNetV3 on CIFAR-100 with custom log dir
```bash
python train.py \
  --dataset cifar100 \
  --model_name mobilenetv3_l \
  --batch_size 128 \
  --num_epochs 100 \
  --use_tensorboard \
  --tensorboard_log_dir experiments/cifar100
```

### Compare different optimizers
```bash
# Run 1: Adam
python train.py --config config/resnet18_adam.json --use_tensorboard

# Run 2: AdamW
python train.py --config config/resnet18_adamw.json --use_tensorboard

# Run 3: SGD
python train.py --config config/resnet18_sgd.json --use_tensorboard

# View all runs
tensorboard --logdir=logs
```

## Tips

1. **Real-time monitoring**: Start TensorBoard before training to watch metrics in real-time
2. **Port conflicts**: If port 6006 is busy, use: `tensorboard --logdir=logs --port=6007`
3. **Remote access**: For remote servers, use port forwarding:
   ```bash
   ssh -L 6006:localhost:6006 user@remote-server
   ```
4. **Clean old logs**: Remove unwanted experiments from the logs folder to declutter TensorBoard
5. **Hyperparameter comparison**: Use the HPARAMS tab to compare runs with different configurations

## Troubleshooting

### "No dashboards are active"
- Make sure you're pointing to the correct log directory
- Wait a few seconds for logs to be written

### Port already in use
```bash
tensorboard --logdir=logs --port=6007
```

### Can't access from browser
- Check if TensorBoard is running: `ps aux | grep tensorboard`
- Verify firewall settings
- Try accessing via `127.0.0.1:6006` instead of `localhost:6006`

## Integration with Existing Tools

TensorBoard works alongside:
- ✅ **W&B**: Use both for cloud backup and local monitoring
- ✅ **Training history plots**: PNG files still saved to results folder
- ✅ **JSON history**: NPZ files still saved for programmatic access
- ✅ **Model checkpoints**: Best models still saved to results folder

---

**Last Updated:** December 18, 2025
