#!/bin/bash
# 消融实验 (利用6张GPU并行跑不同配置)
set -e

export PYTHONPATH="${PYTHONPATH}:./src"
DATASET_CONFIG="configs/dataset/skippd.yaml"
OUTPUT="outputs/ablation"

echo "========================================="
echo "Running Ablation Studies"
echo "========================================="

# 方案A消融: 验证各模块贡献
# A-abl1: 无DCN (用普通Conv解码器)
# A-abl2: 无Mamba (用标准LSTM时间模块)
# A-abl3: 无双头 (仅云图重建)

# 方案B消融:
# B-abl1: 无AFNO
# B-abl2: 无PhyCell
# B-abl3: 无多尺度

# 注: 消融配置需要对应创建 (从默认配置修改)
# 这里展示并行执行框架

CONFIGS=(
    "configs/scheme_a/abl_no_dcn.yaml"
    "configs/scheme_a/abl_no_mamba.yaml"
    "configs/scheme_a/abl_single_head.yaml"
    "configs/scheme_b/abl_no_afno.yaml"
    "configs/scheme_b/abl_no_phycell.yaml"
    "configs/scheme_c/abl_no_mask_guide.yaml"
)

GPU_ID=0
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$cfg" ]; then
        echo "[GPU $GPU_ID] Running ablation: $cfg"
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/train.py \
            --config "$cfg" \
            --dataset_config $DATASET_CONFIG \
            --output_dir $OUTPUT \
            --no_wandb &
        GPU_ID=$(( (GPU_ID + 1) % 6 ))
    else
        echo "Config not found: $cfg (skip)"
    fi
done

wait
echo "All ablation experiments complete!"
