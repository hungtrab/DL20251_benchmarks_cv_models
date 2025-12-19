#!/bin/bash

# ConvNeXtV2 Experiments Script
# This script runs experiments for ConvNeXtV2 fine-tuning with 3 scenarios:
# 1. No pretrain - train from scratch
# 2. Self-dataset pretrain - pretrain on same dataset, then fine-tune
# 3. TIMM checkpoint pretrain - use pretrained weights from timm, then fine-tune
# Each scenario is tested with both constant and cosine learning rate schedulers
# Experiments are run across multiple datasets: mnist, fashionmnist, intel, mit, cifar100, caltech101

# Configuration
MODEL="convnextv2_t"
OPTIMIZER="adamw"
LR=0.001
BATCH_SIZE=32
EPOCHS_PRETRAIN=1
EPOCHS_FINETUNE=1
DROPOUT=0.1
SEED=42

# W&B and TensorBoard flags
WANDB_PROJECT="convnext-experiments"
USE_WANDB="--use_wandb"
USE_TB="--use_tensorboard"

# TIMM checkpoint path (download if needed)
TIMM_CHECKPOINT="checkpoints/convnextv2_tiny_1k_224_fcmae.pt"

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints

# Download TIMM checkpoint if not present
if [ ! -f "$TIMM_CHECKPOINT" ]; then
    echo "Downloading TIMM ConvNeXtV2-Tiny checkpoint..."
    wget -P checkpoints https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt
fi

# Dataset configurations: name, input_size
declare -a DATASETS=(
    "mnist:28"
    "fashionmnist:28"
    "intel:150"
    "mit:224"
    "cifar100:32"
    "caltech101:224"
)

# Learning rate schedulers
declare -a SCHEDULERS=("constant" "cosine")

echo "================================================"
echo "ConvNeXtV2 Fine-tuning Experiments"
echo "================================================"
echo "Model: $MODEL"
echo "Optimizer: $OPTIMIZER"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Pretrain Epochs: $EPOCHS_PRETRAIN"
echo "Fine-tune Epochs: $EPOCHS_FINETUNE"
echo "Datasets: ${DATASETS[@]}"
echo "Schedulers: ${SCHEDULERS[@]}"
echo "================================================"

# Loop through all datasets
for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET INPUT_SIZE <<< "$dataset_config"
    
    echo ""
    echo "========================================"
    echo "Processing Dataset: $DATASET"
    echo "Input Size: ${INPUT_SIZE}x${INPUT_SIZE}"
    echo "========================================"
    
    # Loop through schedulers
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Scheduler: $SCHEDULER"
        echo "----------------------------------------"
        
        # Experiment 1: No pretrain - train from scratch
        echo ""
        echo "[Experiment 1/$((2*${#SCHEDULERS[@]}))] No Pretrain (Train from Scratch)"
        python train.py \
            --dataset_name "$DATASET" \
            --model_name "$MODEL" \
            --optimizer "$OPTIMIZER" \
            --scheduler "$SCHEDULER" \
            --learning_rate $LR \
            --batch_size $BATCH_SIZE \
            --num_epochs $EPOCHS_FINETUNE \
            --dropout_rate $DROPOUT \
            --seed $SEED \
            --input_size $INPUT_SIZE \
            $USE_WANDB \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "${DATASET}_${MODEL}_no-pretrain_${SCHEDULER}" \
            $USE_TB
        
        # Experiment 2: Self-dataset pretrain + fine-tune
        echo ""
        echo "[Experiment 2/$((2*${#SCHEDULERS[@]}))] Self-Dataset Pretrain + Fine-tune"
        
        # Define checkpoint path for self-pretrained model
        SELF_CHECKPOINT="checkpoints/${DATASET}_${MODEL}_pretrain.pth"
        
        # Check if self-pretrained checkpoint exists
        if [ ! -f "$SELF_CHECKPOINT" ]; then
            echo "Self-pretrained checkpoint not found. Skipping self-pretrain experiment for $DATASET."
            echo "To run this experiment, first pretrain the model on $DATASET and save it to $SELF_CHECKPOINT"
        else
            echo "Using self-pretrained checkpoint: $SELF_CHECKPOINT"
            python train.py \
                --dataset_name "$DATASET" \
                --model_name "$MODEL" \
                --optimizer "$OPTIMIZER" \
                --scheduler "$SCHEDULER" \
                --learning_rate $LR \
                --batch_size $BATCH_SIZE \
                --num_epochs $EPOCHS_FINETUNE \
                --dropout_rate $DROPOUT \
                --seed $SEED \
                --input_size $INPUT_SIZE \
                --pretrained_path "$SELF_CHECKPOINT" \
                $USE_WANDB \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_run_name "${DATASET}_${MODEL}_self-pretrain_${SCHEDULER}" \
                $USE_TB
        fi
        
        # Experiment 3: TIMM checkpoint pretrain + fine-tune
        echo ""
        echo "[Experiment 3/$((2*${#SCHEDULERS[@]}))] TIMM Checkpoint Pretrain + Fine-tune"
        python train.py \
            --dataset_name "$DATASET" \
            --model_name "$MODEL" \
            --optimizer "$OPTIMIZER" \
            --scheduler "$SCHEDULER" \
            --learning_rate $LR \
            --batch_size $BATCH_SIZE \
            --num_epochs $EPOCHS_FINETUNE \
            --dropout_rate $DROPOUT \
            --seed $SEED \
            --input_size $INPUT_SIZE \
            --pretrained_path "$TIMM_CHECKPOINT" \
            $USE_WANDB \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "${DATASET}_${MODEL}_timm-pretrain_${SCHEDULER}" \
            $USE_TB
        
        echo ""
        echo "Completed experiments for $DATASET with $SCHEDULER scheduler"
    done
    
    echo ""
    echo "Completed all experiments for $DATASET"
    echo "========================================"
done

echo ""
echo "================================================"
echo "All ConvNeXtV2 experiments completed!"
echo "================================================"
echo "Total experiments run: $((${#DATASETS[@]} * ${#SCHEDULERS[@]} * 3))"
echo "Check results in:"
echo "  - W&B Project: $WANDB_PROJECT"
echo "  - TensorBoard: logs/"
echo "  - Saved models: results/"
echo "================================================"
