#!/bin/bash
# 跨域泛化测试: 在 NREL TSI-880 上评估 SKIPP'D 训练的模型
set -e

export PYTHONPATH="${PYTHONPATH}:./src"
DATASET_CONFIG="configs/dataset/nrel_tsi.yaml"
OUTPUT="outputs"

echo "========================================="
echo "Cross-Domain Evaluation (NREL TSI-880)"
echo "Models trained on SKIPP'D, tested on NREL"
echo "========================================="

MODELS=(
    "convlstm"
    "predrnn"
    "simvp"
    "simvp_tau"
    "scheme_a"
    "scheme_b"
    "scheme_c"
)

GPU_ID=0
for model in "${MODELS[@]}"; do
    CKPT="$OUTPUT/$model/best.pth"
    if [ -f "$CKPT" ]; then
        echo "[GPU $GPU_ID] Evaluating $model on NREL..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/evaluate.py \
            --config "configs/baselines/${model}.yaml" \
            --dataset_config $DATASET_CONFIG \
            --checkpoint "$CKPT" \
            --output_dir "$OUTPUT/cross_domain/$model" &
        GPU_ID=$(( (GPU_ID + 1) % 6 ))
    else
        echo "Checkpoint not found for $model: $CKPT (skip)"
    fi
done

wait
echo "Cross-domain evaluation complete!"
echo "Results saved to $OUTPUT/cross_domain/"
