#!/bin/bash

# ============================================================================
# Comprehensive Training Script for CNN Benchmark
# 9 Models × 4 Datasets = 36 Experiments
# ============================================================================

set -e  # Exit on error

# ============================================================================
# 1. AlexNet (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training AlexNet =========="

python train.py --dataset cifar100 --model_name alexnet --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name alexnet --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name alexnet --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name alexnet --num_epochs 200 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 2. VGG16-BN (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training VGG16-BN =========="

python train.py --dataset cifar100 --model_name vgg16_bn --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name vgg16_bn --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name vgg16_bn --num_epochs 100 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name vgg16_bn --num_epochs 200 --config config/hpo_new/group_legacy_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 3. ResNet18 (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training ResNet18 =========="

python train.py --dataset cifar100 --model_name resnet18 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name resnet18 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name resnet18 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name resnet18 --num_epochs 200 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 4. ResNet34 (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training ResNet34 =========="

python train.py --dataset cifar100 --model_name resnet34 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name resnet34 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name resnet34 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name resnet34 --num_epochs 200 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 5. ResNet50 (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training ResNet50 =========="

python train.py --dataset cifar100 --model_name resnet50 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name resnet50 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name resnet50 --num_epochs 100 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name resnet50 --num_epochs 200 --config config/hpo_new/group_resnet_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 6. EfficientNetV2-M (250 epochs for cifar100, caltech101, intel; 300 for mit)
# ============================================================================
echo "========== Training EfficientNetV2-M =========="

python train.py --dataset cifar100 --model_name efficientnetv2_m --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name efficientnetv2_m --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name efficientnetv2_m --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name efficientnetv2_m --num_epochs 300 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 7. EfficientNetV2-L (250 epochs for cifar100, caltech101, intel; 300 for mit)
# ============================================================================
echo "========== Training EfficientNetV2-L =========="

python train.py --dataset cifar100 --model_name efficientnetv2_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name efficientnetv2_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name efficientnetv2_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name efficientnetv2_l --num_epochs 300 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 8. MobileNetV3-L (250 epochs for cifar100, caltech101, intel; 300 for mit)
# ============================================================================
echo "========== Training MobileNetV3-L =========="

python train.py --dataset cifar100 --model_name mobilenetv3_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset intel --model_name mobilenetv3_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset caltech101 --model_name mobilenetv3_l --num_epochs 250 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

python train.py --dataset mit --model_name mobilenetv3_l --num_epochs 300 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training

# ============================================================================
# 9. ConvNeXtV2-T with Self-Supervised Pretraining
#    (100 epochs for cifar100, caltech101, intel; 200 for mit)
# ============================================================================
echo "========== Training ConvNeXtV2-T (with FCMAE Pretraining) =========="

python train.py --dataset cifar100 --model_name convnextv2_t --num_epochs 100 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training --conv_pretrain --conv_pretrain_epochs 50 --conv_mask_ratio 0.6 --conv_pretrain_lr 1.5e-4 --conv_pretrain_wd 0.05

python train.py --dataset intel --model_name convnextv2_t --num_epochs 100 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training --conv_pretrain --conv_pretrain_epochs 50 --conv_mask_ratio 0.6 --conv_pretrain_lr 1.5e-4 --conv_pretrain_wd 0.05

python train.py --dataset caltech101 --model_name convnextv2_t --num_epochs 100 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training --conv_pretrain --conv_pretrain_epochs 50 --conv_mask_ratio 0.6 --conv_pretrain_lr 1.5e-4 --conv_pretrain_wd 0.05

python train.py --dataset mit --model_name convnextv2_t --num_epochs 200 --config config/hpo_new/group_modern_best.json --use_wandb --wandb_project dl20251 --input_size 224 --batch_size 64 --val --val_size 0.2 --eval_full --use_mixup --mixup_alpha 0.2 --use_cutmix --cutmix_alpha 1.0 --cutmix_prob 0.5 --use_sam --sam_rho 0.05 --label_smoothing 0.1 --adaptive_training --conv_pretrain --conv_pretrain_epochs 50 --conv_mask_ratio 0.6 --conv_pretrain_lr 1.5e-4 --conv_pretrain_wd 0.05

# ============================================================================
# All experiments complete!
# ============================================================================
echo "========================================="
echo "All 36 experiments completed successfully!"
echo "========================================="
