"""
Cuboid Attention 模块
参考 Earthformer (NeurIPS 2022)，将时空数据分解为立方体块进行高效注意力计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class CuboidSelfAttention(nn.Module):
    """
    Cuboid Self-Attention: 将时空体 [T,H,W] 划分为立方体，
    在 cuboid 内部做 local attention，cuboid 间做 global attention
    """

    def __init__(self, d_model, num_heads=8, cuboid_size=(2, 4, 4),
                 attn_drop=0.0, proj_drop=0.0):
        """
        Args:
            d_model: 特征维度
            num_heads: 注意力头数
            cuboid_size: (T_c, H_c, W_c) 立方体大小
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.cuboid_size = cuboid_size

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W, D]
        Returns:
            [B, T, H, W, D]
        """
        B, T, H, W, D = x.shape
        Tc, Hc, Wc = self.cuboid_size

        # Padding if needed
        pad_t = (Tc - T % Tc) % Tc
        pad_h = (Hc - H % Hc) % Hc
        pad_w = (Wc - W % Wc) % Wc
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

        Tp, Hp, Wp = T + pad_t, H + pad_h, W + pad_w

        # 划分为 cuboids
        nT, nH, nW = Tp // Tc, Hp // Hc, Wp // Wc
        x = rearrange(x, 'b (nT Tc) (nH Hc) (nW Wc) d -> (b nT nH nW) (Tc Hc Wc) d',
                       Tc=Tc, Hc=Hc, Wc=Wc)

        # 在每个 cuboid 内做 self-attention
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(-1, Tc * Hc * Wc, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, Tc * Hc * Wc, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + residual

        # 恢复形状
        x = rearrange(x, '(b nT nH nW) (Tc Hc Wc) d -> b (nT Tc) (nH Hc) (nW Wc) d',
                       b=B, nT=nT, nH=nH, nW=nW, Tc=Tc, Hc=Hc, Wc=Wc)

        # 去除 padding
        x = x[:, :T, :H, :W, :]
        return x


class CuboidTransformerBlock(nn.Module):
    """Cuboid Transformer Block: Cuboid Attention + FFN"""

    def __init__(self, d_model, num_heads=8, cuboid_size=(2, 4, 4),
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.attn = CuboidSelfAttention(
            d_model, num_heads, cuboid_size,
            attn_drop=drop, proj_drop=drop,
        )
        self.norm = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """x: [B, T, H, W, D]"""
        x = self.attn(x)
        residual = x
        x = self.norm(x)
        x = residual + self.mlp(x)
        return x
