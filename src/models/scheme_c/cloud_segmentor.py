"""
云分割前置网络
轻量 U-Net 云分割器，用于提取云 mask 序列作为扩散模型的条件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class CloudSegmentor(nn.Module):
    """
    轻量 U-Net 云分割器
    输入单帧天空图像，输出云/非云二值 mask
    可用 SWIMSEG + SWINySEG 预训练
    """

    def __init__(self, in_channels=3, base_dim=32):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]

        # Encoder
        self.enc1 = DoubleConv(in_channels, dims[0])
        self.enc2 = DoubleConv(dims[0], dims[1])
        self.enc3 = DoubleConv(dims[1], dims[2])
        self.enc4 = DoubleConv(dims[2], dims[3])

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.dec3 = DoubleConv(dims[2] * 2, dims[2])
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.dec2 = DoubleConv(dims[1] * 2, dims[1])
        self.up1 = nn.ConvTranspose2d(dims[1], dims[0], 2, stride=2)
        self.dec1 = DoubleConv(dims[0] * 2, dims[0])

        # 分割头
        self.head = nn.Conv2d(dims[0], 1, 1)

        # 特征提取 (用于条件注入扩散模型)
        self.feat_proj = nn.Conv2d(dims[0], dims[1], 1)

    def forward(self, x, return_features=False):
        """
        Args:
            x: [B, C, H, W]
            return_features: 是否返回中间特征 (用于扩散模型条件)
        Returns:
            mask: [B, 1, H, W] logits
            features: [B, D, H, W] (optional)
        """
        # Encoder
        e1 = self.enc1(x)           # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))   # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))   # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))   # [B, 256, H/8, W/8]

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        mask = self.head(d1)

        if return_features:
            feat = self.feat_proj(d1)
            return mask, feat
        return mask

    def segment_sequence(self, frames, return_features=False):
        """
        分割整个序列
        Args:
            frames: [B, T, C, H, W]
        Returns:
            masks: [B, T, 1, H, W]
            features: [B, T, D, H, W] (optional)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)

        if return_features:
            masks, feats = self.forward(frames_flat, return_features=True)
            masks = masks.reshape(B, T, 1, H, W)
            feats = feats.reshape(B, T, -1, H, W)
            return masks, feats

        masks = self.forward(frames_flat, return_features=False)
        masks = masks.reshape(B, T, 1, H, W)
        return masks
