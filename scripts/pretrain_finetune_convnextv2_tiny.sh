export CUDA_VISIBLE_DEVICES=2
python ../train.py \
  --dataset intel \
  --model_name convnextv2_tiny \
  --checkpoint_path checkpoints_official/convnextv2_tiny_1k_224.pth \
  --input_size 224 \
  --batch_size 64 \
  --num_epochs 20 \
  --learning_rate 5e-5 \
  --optimizer adamw \
  --scheduler cosine \
  --num_warmup_steps 100 \
  --seed 42