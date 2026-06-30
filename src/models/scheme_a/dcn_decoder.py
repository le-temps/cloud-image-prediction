"""
可变形卷积解码器 (DCN v2 Decoder)
自适应学习采样位置，捕捉云团非刚性变形运动
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmcv.ops import DeformConv2d, deform_conv2d
    HAS_DCN = True
except ImportError:
    HAS_DCN = False


class DeformableConvBlock(nn.Module):
    """可变形卷积块 with offset prediction"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, deform_groups=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups

        # Offset prediction: 每个deform group有 2*k*k 个offset
        offset_channels = 2 * kernel_size * kernel_size * deform_groups
        # Mask prediction (DCN v2): 每个deform group有 k*k 个mask
        mask_channels = kernel_size * kernel_size * deform_groups

        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels,
            kernel_size=3, padding=1
        )
        self.mask_conv = nn.Conv2d(
            in_channels, mask_channels,
            kernel_size=3, padding=1
        )

        if HAS_DCN:
            self.dcn = DeformConv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                deform_groups=deform_groups,
            )
        else:
            # Fallback: 标准卷积
            self.dcn = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

        # 初始化 offset 为 0
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C_out, H, W]
        """
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        if HAS_DCN:
            out = self.dcn(x, offset)
        else:
            out = self.dcn(x)

        out = self.act(self.norm(out))
        return out


class DCNDecoder(nn.Module):
    """
    可变形卷积解码器
    多层 DCN + 上采样，逐步恢复空间分辨率
    """

    def __init__(self, in_dim=256, out_channels=3, num_layers=4,
                 deform_groups=4, patch_size=4):
        super().__init__()
        self.num_layers = num_layers
        self.patch_size = patch_size

        dims = [in_dim // (2 ** i) for i in range(num_layers)]
        dims = [max(d, 32) for d in dims]  # 最小32

        layers = []
        for i in range(num_layers):
            in_ch = dims[i] if i < len(dims) else dims[-1]
            out_ch = dims[i + 1] if i + 1 < len(dims) else dims[-1]
            layers.append(DeformableConvBlock(
                in_ch, out_ch,
                deform_groups=deform_groups,
            ))
        self.layers = nn.ModuleList(layers)

        # 上采样到原始分辨率
        self.upsample = nn.ConvTranspose2d(
            dims[-1], dims[-1],
            kernel_size=patch_size, stride=patch_size,
        )

        # 最终输出头
        self.head = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

    def forward(self, x, skip_features=None):
        """
        Args:
            x: [B, D, H_p, W_p] encoder 输出 (patch space)
            skip_features: list of skip connection features (optional)
        Returns:
            [B, C_out, H, W] 原始分辨率输出
        """
        for i, layer in enumerate(self.layers):
            if skip_features and i < len(skip_features):
                x = x + skip_features[-(i + 1)]
            x = layer(x)

        # 上采样
        x = self.upsample(x)
        out = self.head(x)
        return out
