"""Patch Embedding 模块"""
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """标准 Patch Embedding"""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, N, D] where N = (H/P)*(W/P)
        """
        x = self.proj(x)  # [B, D, H/P, W/P]
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x, (Hp, Wp)


class PatchEmbed2D(nn.Module):
    """2D Patch Embedding 保持空间结构"""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, D, H/P, W/P]
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class MultiScalePatchEmbed(nn.Module):
    """多尺度 Patch Embedding (方案B使用)"""

    def __init__(self, in_channels=3, embed_dim=256, patch_sizes=[2, 4, 8]):
        super().__init__()
        self.num_scales = len(patch_sizes)
        dim_per_scale = embed_dim // self.num_scales

        self.embeds = nn.ModuleList([
            nn.Conv2d(in_channels, dim_per_scale,
                      kernel_size=ps, stride=ps)
            for ps in patch_sizes
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, dim_per_scale)
            for _ in patch_sizes
        ])
        self.patch_sizes = patch_sizes

    def forward(self, x):
        """
        Returns list of multi-scale features
        """
        features = []
        for embed, norm in zip(self.embeds, self.norms):
            feat = norm(embed(x))
            features.append(feat)
        return features


class PatchRecover(nn.Module):
    """从 patch 特征恢复为图像"""

    def __init__(self, embed_dim=256, out_channels=3, patch_size=4):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels,
                                       kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        """
        Args:
            x: [B, N, D] or [B, D, Hp, Wp]
            H, W: 原始空间大小 (patch之后的)
        """
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.transpose(1, 2).reshape(B, D, H, W)
        return self.proj(x)
