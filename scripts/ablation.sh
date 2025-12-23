#!/bin/bash
cd ..

# ==========================================
# GLOBAL SETTINGS (Cấu hình chung)
# ==========================================
DATASET="cifar100_224"  # Dùng bộ data đã resize 224 để tận dụng Pretrained tốt nhất
INPUT_SIZE=224
BATCH_SIZE=128
NUM_EPOCHS=50           # 30-50 epochs là đủ cho Fine-tuning
WARMUP_STEPS=300        # ~1 epoch warm up
PROJECT_NAME="dl20251-ablation-study" # Tên project trên WandB

# Hàm chạy training để code gọn hơn
run_train() {
    local model=$1
    local optimizer=$2
    local lr=$3
    local dropout=$4
    local run_name=$5
    local extra_args=$6
    local scheduler=$7

    echo "----------------------------------------------------------------"
    echo "STARTING: $run_name"
    echo "Model: $model | Optim: $optimizer | LR: $lr | Drop: $dropout"
    echo "----------------------------------------------------------------"

    python train.py \
        --dataset $DATASET \
        --model_name $model \
        --input_size $INPUT_SIZE \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $lr \
        --optimizer $optimizer \
        --scheduler $scheduler \
        --num_warmup_steps $WARMUP_STEPS \
        --dropout_rate $dropout \
        --pretrained \
        --fasttrain \
        --use_wandb \
        --wandb_project $PROJECT_NAME \
        --wandb_run_name "$run_name" \
        --use_tensorboard \
        --tensorboard_log_dir "results/tb/$run_name" \
        $extra_args

    if [ $? -eq 0 ]; then
        echo "✓ SUCCESS: $run_name"
    else
        echo "✗ FAILED: $run_name"
        exit 1 # Dừng script nếu lỗi để check
    fi
    echo ""
}

# ==========================================
# EXPERIMENT 1: OPTIMIZER COMPARISON (Baseline: ResNet18)
# So sánh AdamW vs Adam vs SGD
# ==========================================
echo "=== GROUP 1: OPTIMIZER ABLATION ==="

# 1.1 Baseline (AdamW)
run_train "resnet18" "adamw" 0.001 0.0 "Exp1_ResNet18_AdamW_Baseline" "" "cosine"

# 1.2 Adam (Thường Weight Decay = 0 hoặc thấp hơn AdamW)
run_train "resnet18" "adam" 0.001 0.0 "Exp1_ResNet18_Adam_Legacy" "" "cosine"

# 1.3 SGD (Momentum thường cần LR cao hơn, ví dụ 0.01 hoặc 0.005)
# Lưu ý: Code train.py cần support momentum mặc định cho SGD
run_train "resnet18" "sgd" 0.01 0.0 "Exp1_ResNet18_SGD_HighLR" "" "cosine"


# ==========================================
# EXPERIMENT 2: DROPOUT RATE (Model: VGG16_BN)
# VGG có nhiều params ở FC layer nên nhạy với Dropout
# ==========================================
echo "=== GROUP 2: DROPOUT ABLATION ==="

# 2.1 No Dropout
run_train "vgg16_bn" "adamw" 0.0001 0.0 "Exp2_VGG16BN_Drop0.0" ""

# 2.2 Dropout 0.2
run_train "vgg16_bn" "adamw" 0.0001 0.2 "Exp2_VGG16BN_Drop0.2" ""

# 2.3 Dropout 0.5 (Mặc định của VGG gốc)
run_train "vgg16_bn" "adamw" 0.0001 0.5 "Exp2_VGG16BN_Drop0.5" ""


# ==========================================
# EXPERIMENT 3: SCHEDULER COMPARISON (Model: ResNet18)
# So sánh Constant vs Cosine Annealing
# ==========================================
echo "=== GROUP 3: SCHEDULER ABLATION ==="
# 3.1 Constant LR
run_train "resnet18" "adamw" 0.001 0.0 "Exp3_ResNet18_ConstantLR" "" "constant"
# 3.2 Cosine Annealing LR
run_train "resnet18" "adamw" 0.001 0.0 "Exp3_ResNet18_CosineLR" "" "cosine"
# 3.3 Linear Decay LR
run_train "resnet18" "adamw" 0.001 0.0 "Exp3_ResNet18_LinearLR" "" "linear"


echo "All ablation studies completed!"