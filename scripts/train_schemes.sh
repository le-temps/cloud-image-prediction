#!/bin/bash
# 训练三个创新方案 (各用2张GPU DDP)
set -e

export PYTHONPATH="${PYTHONPATH}:./src"
DATASET_CONFIG="configs/dataset/skippd.yaml"
OUTPUT="outputs"

echo "========================================="
echo "Training 3 Schemes (2 GPUs each, DDP)"
echo "========================================="

# 方案A: CloudMambaNet (GPU 0-1)
echo "[GPU 0-1] Training Scheme A: CloudMambaNet..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
    src/train.py \
    --config configs/scheme_a/default.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT &
PID_A=$!

# 方案B: CloudPhysicsNet (GPU 2-3)
echo "[GPU 2-3] Training Scheme B: CloudPhysicsNet..."
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
    src/train.py \
    --config configs/scheme_b/default.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT &
PID_B=$!

# 方案C: CloudDiffNet (GPU 4-5)
echo "[GPU 4-5] Training Scheme C: CloudDiffNet..."
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29502 \
    src/train.py \
    --config configs/scheme_c/default.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT &
PID_C=$!

echo ""
echo "PIDs: A=$PID_A, B=$PID_B, C=$PID_C"
echo "Monitoring..."
wait
echo "All schemes training complete!"
