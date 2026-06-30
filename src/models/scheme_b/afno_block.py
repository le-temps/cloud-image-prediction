"""
AFNO Block - Adaptive Fourier Neural Operator
在频率域进行空间混合，替代标准自注意力的 token mixing
参考 FourCastNet (NVIDIA)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator Block
    核心思路: FFT → 频域可学习滤波 → IFFT
    """

    def __init__(self, d_model, num_blocks=8, hidden_ratio=1.0, drop=0.0):
        """
        Args:
            d_model: 特征维度
            num_blocks: 频域中可学习参数块数 (类似 channel groups)
            hidden_ratio: 频域隐藏层扩展比例
        """
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.block_size = d_model // num_blocks
        self.hidden_size = int(self.block_size * hidden_ratio)

        self.norm = nn.LayerNorm(d_model)

        # 频域可学习权重 (复数)
        # 每个 block 有独立的频域 MLP
        self.w1 = nn.Parameter(torch.randn(2, num_blocks, self.block_size, self.hidden_size) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(2, num_blocks, self.hidden_size))
        self.w2 = nn.Parameter(torch.randn(2, num_blocks, self.hidden_size, self.block_size) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(2, num_blocks, self.block_size))

        self.softshrink = nn.Softshrink(lambd=0.01)
        self.drop = nn.Dropout(drop)

    def _complex_mul(self, x_real, x_imag, w_real, w_imag):
        """复数矩阵乘法 (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        out_real = torch.einsum('...i,io->...o', x_real, w_real) - \
                   torch.einsum('...i,io->...o', x_imag, w_imag)
        out_imag = torch.einsum('...i,io->...o', x_real, w_imag) + \
                   torch.einsum('...i,io->...o', x_imag, w_real)
        return out_real, out_imag

    def forward(self, x):
        """
        Args:
            x: [B, H, W, D]
        Returns:
            [B, H, W, D]
        """
        B, H, W, D = x.shape
        residual = x
        x = self.norm(x)

        # 转换为 [B, D, H, W] 便于 FFT
        x = x.permute(0, 3, 1, 2)

        # 2D FFT
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        x_real = x_fft.real  # [B, D, H, W//2+1]
        x_imag = x_fft.imag

        # 在频域做可学习滤波 (分 block 处理)
        x_real = x_real.reshape(B, self.num_blocks, self.block_size, H, -1)
        x_imag = x_imag.reshape(B, self.num_blocks, self.block_size, H, -1)

        # 移动维度以便矩阵乘法: [B, num_blocks, H, W_fft, block_size]
        x_real = x_real.permute(0, 1, 3, 4, 2)
        x_imag = x_imag.permute(0, 1, 3, 4, 2)

        # 第一层 MLP
        h_real, h_imag = self._complex_mul(
            x_real, x_imag, self.w1[0], self.w1[1]
        )
        h_real = h_real + self.b1[0]
        h_imag = h_imag + self.b1[1]

        # 非线性激活 (Softshrink on magnitude)
        h_real = self.softshrink(h_real)
        h_imag = self.softshrink(h_imag)

        # 第二层 MLP
        o_real, o_imag = self._complex_mul(
            h_real, h_imag, self.w2[0], self.w2[1]
        )
        o_real = o_real + self.b2[0]
        o_imag = o_imag + self.b2[1]

        # 恢复形状 [B, D, H, W_fft]
        o_real = o_real.permute(0, 1, 4, 2, 3).reshape(B, D, H, -1)
        o_imag = o_imag.permute(0, 1, 4, 2, 3).reshape(B, D, H, -1)

        # IFFT
        x_fft_out = torch.complex(o_real, o_imag)
        x_out = torch.fft.irfft2(x_fft_out, s=(H, W), dim=(-2, -1), norm='ortho')

        # 转回 [B, H, W, D]
        x_out = x_out.permute(0, 2, 3, 1)
        x_out = self.drop(x_out)

        return x_out + residual
