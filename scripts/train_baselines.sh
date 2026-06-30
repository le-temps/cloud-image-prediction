#!/bin/bash
# 批量训练基线模型 (利用6张GPU并行)
set -e

export PYTHONPATH="${PYTHONPATH}:./src"
DATASET_CONFIG="configs/dataset/skippd.yaml"
OUTPUT="outputs"

echo "========================================="
echo "Training Baselines (6 models in parallel)"
echo "========================================="

# GPU 0: ConvLSTM
echo "[GPU 0] Training ConvLSTM..."
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --config configs/baselines/convlstm.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT \
    --no_wandb &

# GPU 1: PredRNN++
echo "[GPU 1] Training PredRNN++..."
CUDA_VISIBLE_DEVICES=1 python src/train.py \
    --config configs/baselines/predrnn.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT \
    --no_wandb &

# GPU 2: SimVP
echo "[GPU 2] Training SimVP..."
CUDA_VISIBLE_DEVICES=2 python src/train.py \
    --config configs/baselines/simvp.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT \
    --no_wandb &

# GPU 3: SimVP + TAU
echo "[GPU 3] Training SimVP+TAU..."
CUDA_VISIBLE_DEVICES=3 python src/train.py \
    --config configs/baselines/simvp_tau.yaml \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT \
    --no_wandb &

# GPU 4-5: 预留给 STANet / MSAN (需要单独实现)
# echo "[GPU 4] Training STANet..."
# echo "[GPU 5] Training MSAN..."

echo ""
echo "All baseline jobs launched. Monitoring..."
wait
echo "All baselines training complete!"
