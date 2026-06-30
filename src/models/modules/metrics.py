"""评估指标模块"""
import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips


class MetricCalculator:
    """统一的评估指标计算器"""

    def __init__(self, device='cuda'):
        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()

    @torch.no_grad()
    def compute_all(self, pred, target, cloud_mask_pred=None, cloud_mask_target=None):
        """
        计算所有指标
        Args:
            pred: [B, T, C, H, W] 预测图像
            target: [B, T, C, H, W] 真值图像
            cloud_mask_pred: [B, T, 1, H, W] 预测云mask (optional)
            cloud_mask_target: [B, T, 1, H, W] 真值云mask (optional)
        Returns:
            dict of metrics
        """
        B, T, C, H, W = pred.shape
        # 展平时间维度用于计算
        pred_flat = pred.reshape(B * T, C, H, W)
        target_flat = target.reshape(B * T, C, H, W)

        metrics = {}

        # 图像质量指标
        metrics['mse'] = F.mse_loss(pred_flat, target_flat).item()
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['ssim'] = self.ssim(pred_flat, target_flat).item()
        metrics['psnr'] = self.psnr(pred_flat, target_flat).item()

        # LPIPS (需要 [-1, 1] 范围)
        pred_lpips = pred_flat * 2 - 1
        target_lpips = target_flat * 2 - 1
        metrics['lpips'] = self.lpips_fn(pred_lpips, target_lpips).mean().item()

        # 云预测专用指标
        if cloud_mask_pred is not None and cloud_mask_target is not None:
            cloud_metrics = self._compute_cloud_metrics(
                cloud_mask_pred, cloud_mask_target
            )
            metrics.update(cloud_metrics)

        return metrics

    @torch.no_grad()
    def compute_per_frame(self, pred, target):
        """
        逐帧计算指标，用于分析预测时间步的性能衰减
        Args:
            pred: [B, T, C, H, W]
            target: [B, T, C, H, W]
        Returns:
            dict of lists (每个时间步一个值)
        """
        B, T, C, H, W = pred.shape
        frame_metrics = {'ssim': [], 'psnr': [], 'mse': []}

        for t in range(T):
            p = pred[:, t]
            g = target[:, t]
            frame_metrics['ssim'].append(self.ssim(p, g).item())
            frame_metrics['psnr'].append(self.psnr(p, g).item())
            frame_metrics['mse'].append(F.mse_loss(p, g).item())

        return frame_metrics

    def _compute_cloud_metrics(self, pred_mask, target_mask):
        """
        计算云检测相关指标
        Args:
            pred_mask: [B, T, 1, H, W] 概率值或二值
            target_mask: [B, T, 1, H, W] 二值
        """
        # 二值化
        pred_bin = (pred_mask > 0.5).float()
        target_bin = target_mask.float()

        # 展平
        pred_flat = pred_bin.reshape(-1)
        target_flat = target_bin.reshape(-1)

        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        return {
            'cloud_accuracy': accuracy.item(),
            'cloud_precision': precision.item(),
            'cloud_recall': recall.item(),
            'cloud_f1': f1.item(),
            'cloud_iou': iou.item(),
        }

    @torch.no_grad()
    def forecast_skill(self, pred, target, persistence):
        """
        Forecast Skill: 相对于 Smart Persistence 的提升
        FS = 1 - MSE(pred) / MSE(persistence)
        """
        mse_pred = F.mse_loss(pred, target).item()
        mse_persist = F.mse_loss(persistence, target).item()
        if mse_persist < 1e-8:
            return 0.0
        return 1.0 - mse_pred / mse_persist
