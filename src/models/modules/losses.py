"""损失函数模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class SSIMLoss(nn.Module):
    """SSIM 损失 (1 - SSIM)"""

    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel

    def _gaussian_window(self, window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]
        """
        B, C, H, W = pred.shape
        window = self._gaussian_window(self.window_size).to(pred.device)
        window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=C)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=C) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class LPIPSLoss(nn.Module):
    """LPIPS 感知损失"""

    def __init__(self, net='alex'):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net)
        self.loss_fn.eval()
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        """输入范围 [0,1]，内部转换到 [-1,1]"""
        return self.loss_fn(pred * 2 - 1, target * 2 - 1).mean()


class CombinedLoss(nn.Module):
    """通用组合损失函数（方案A/B共用）"""

    def __init__(self, mse_weight=1.0, ssim_weight=0.5, lpips_weight=0.1,
                 bce_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.bce_weight = bce_weight

        self.ssim_loss = SSIMLoss()
        if lpips_weight > 0:
            self.lpips_loss = LPIPSLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_img, target_img, pred_mask=None, target_mask=None):
        """
        Args:
            pred_img: [B, T, C, H, W] 预测图像
            target_img: [B, T, C, H, W] 真值图像
            pred_mask: [B, T, 1, H, W] 预测云mask logits (optional)
            target_mask: [B, T, 1, H, W] 真值云mask (optional)
        """
        B, T, C, H, W = pred_img.shape
        pred_flat = pred_img.reshape(B * T, C, H, W)
        target_flat = target_img.reshape(B * T, C, H, W)

        losses = {}

        # MSE
        loss_mse = F.mse_loss(pred_flat, target_flat)
        losses['mse'] = loss_mse
        total = self.mse_weight * loss_mse

        # SSIM
        if self.ssim_weight > 0:
            loss_ssim = self.ssim_loss(pred_flat, target_flat)
            losses['ssim'] = loss_ssim
            total = total + self.ssim_weight * loss_ssim

        # LPIPS
        if self.lpips_weight > 0:
            loss_lpips = self.lpips_loss(pred_flat, target_flat)
            losses['lpips'] = loss_lpips
            total = total + self.lpips_weight * loss_lpips

        # BCE (云mask)
        if self.bce_weight > 0 and pred_mask is not None and target_mask is not None:
            pred_mask_flat = pred_mask.reshape(B * T, 1, H, W)
            target_mask_flat = target_mask.reshape(B * T, 1, H, W)
            loss_bce = self.bce_loss(pred_mask_flat, target_mask_flat)
            losses['bce'] = loss_bce
            total = total + self.bce_weight * loss_bce

        losses['total'] = total
        return total, losses


class PhysicsLoss(nn.Module):
    """物理约束损失：平流方程 ∂u/∂t + v·∇u ≈ 0"""

    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        # Sobel filters for spatial gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred_seq, velocity=None):
        """
        Args:
            pred_seq: [B, T, C, H, W] 预测序列
            velocity: [B, T-1, 2, H, W] 速度场 (optional, 若为None则用相邻帧差估计)
        """
        B, T, C, H, W = pred_seq.shape

        # 时间导数: ∂u/∂t ≈ u(t+1) - u(t)
        du_dt = pred_seq[:, 1:] - pred_seq[:, :-1]  # [B, T-1, C, H, W]

        # 空间梯度
        frames = pred_seq[:, :-1].reshape(B * (T - 1) * C, 1, H, W)
        du_dx = F.conv2d(frames, self.sobel_x.to(frames.device), padding=1)
        du_dy = F.conv2d(frames, self.sobel_y.to(frames.device), padding=1)
        du_dx = du_dx.reshape(B, T - 1, C, H, W)
        du_dy = du_dy.reshape(B, T - 1, C, H, W)

        if velocity is None:
            # 简单估计: 用相邻帧光流近似
            # 平流方程残差: ∂u/∂t 应当尽可能小 (假设慢变化)
            physics_residual = du_dt
        else:
            vx = velocity[:, :, 0:1]  # [B, T-1, 1, H, W]
            vy = velocity[:, :, 1:2]
            advection = vx * du_dx + vy * du_dy
            physics_residual = du_dt + advection

        return self.weight * (physics_residual ** 2).mean()
