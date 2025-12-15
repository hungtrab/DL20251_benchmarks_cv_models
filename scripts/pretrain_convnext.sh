export CUDA_VISIBLE_DEVICES=3
python ../pretrain.py \
  --dataset intel \
  --train_dir /home/vudd/Convnext/DL20251_benchmarks_cv_models/data/intel_image/seg_train/seg_train \
  --output_dir results_pretrain \
  --experiment_name intel_image_pretrain_v1 \
  --model_name fcmae_convnextv2_base \
  --input_size 224 \
  --batch_size 128 \
  --num_epochs 25 \
  --learning_rate 1.5e-4 \
  --weight_decay 0.05 \
  --warmup_epochs 5 \
  --mask_ratio 0.6 \
  --save_freq 10 \
  --num_workers 8 \
  --seed 42