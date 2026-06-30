"""可视化工具"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_prediction(input_seq, pred_seq, target_seq, save_path,
                         cloud_mask_pred=None, cloud_mask_target=None,
                         num_show=5):
    """
    可视化预测结果
    Args:
        input_seq: [T_in, C, H, W] 输入序列
        pred_seq: [T_out, C, H, W] 预测序列
        target_seq: [T_out, C, H, W] 真值序列
        save_path: 保存路径
        cloud_mask_pred: [T_out, 1, H, W] 预测云mask
        cloud_mask_target: [T_out, 1, H, W] 真值云mask
        num_show: 每行显示的帧数
    """
    T_in = input_seq.shape[0]
    T_out = pred_seq.shape[0]

    # 选择均匀间隔的帧
    in_indices = np.linspace(0, T_in - 1, min(num_show, T_in), dtype=int)
    out_indices = np.linspace(0, T_out - 1, min(num_show, T_out), dtype=int)

    has_mask = cloud_mask_pred is not None and cloud_mask_target is not None
    n_rows = 4 if has_mask else 3
    n_cols = num_show

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Row 0: Input
    for i, idx in enumerate(in_indices):
        img = _to_numpy(input_seq[idx])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Input t-{T_in - idx}')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Input', fontsize=12)

    # Row 1: Prediction
    for i, idx in enumerate(out_indices):
        img = _to_numpy(pred_seq[idx])
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Pred t+{idx + 1}')
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel('Prediction', fontsize=12)

    # Row 2: Ground Truth
    for i, idx in enumerate(out_indices):
        img = _to_numpy(target_seq[idx])
        axes[2, i].imshow(img)
        axes[2, i].set_title(f'GT t+{idx + 1}')
        axes[2, i].axis('off')
    axes[2, 0].set_ylabel('Ground Truth', fontsize=12)

    # Row 3: Cloud Mask (if available)
    if has_mask:
        for i, idx in enumerate(out_indices):
            mask_pred = cloud_mask_pred[idx, 0].cpu().numpy()
            mask_gt = cloud_mask_target[idx, 0].cpu().numpy()
            # 红=FP, 绿=TP, 蓝=FN
            overlay = np.stack([
                (mask_pred > 0.5).astype(float) * (mask_gt < 0.5).astype(float),  # FP - red
                (mask_pred > 0.5).astype(float) * (mask_gt > 0.5).astype(float),  # TP - green
                (mask_pred < 0.5).astype(float) * (mask_gt > 0.5).astype(float),  # FN - blue
            ], axis=-1)
            axes[3, i].imshow(overlay)
            axes[3, i].set_title(f'Mask t+{idx + 1}')
            axes[3, i].axis('off')
        axes[3, 0].set_ylabel('Cloud Mask\n(G=TP R=FP B=FN)', fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_over_time(frame_metrics, save_path):
    """绘制指标随预测时间步的变化曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    timesteps = range(1, len(frame_metrics['ssim']) + 1)

    axes[0].plot(timesteps, frame_metrics['ssim'], 'b-o')
    axes[0].set_xlabel('Prediction Step')
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('SSIM vs Prediction Horizon')
    axes[0].grid(True)

    axes[1].plot(timesteps, frame_metrics['psnr'], 'r-o')
    axes[1].set_xlabel('Prediction Step')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR vs Prediction Horizon')
    axes[1].grid(True)

    axes[2].plot(timesteps, frame_metrics['mse'], 'g-o')
    axes[2].set_xlabel('Prediction Step')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('MSE vs Prediction Horizon')
    axes[2].grid(True)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _to_numpy(tensor):
    """将 [C, H, W] tensor 转为可显示的 numpy array"""
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().numpy()
    else:
        img = tensor
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[0] == 1:
        img = img[0]
    img = np.clip(img, 0, 1)
    return img
