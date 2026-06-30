"""
SimVP 基线模型 (含 TAU 变体)
参考 SimVP (CVPR 2022) + TAU (CVPR 2023)
"""
import torch
import torch.nn as nn
from einops import rearrange


class InceptionBlock(nn.Module):
    """SimVP 中的 Inception-style 空间模块"""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.act = nn.GELU()
        self.dw3 = nn.Conv2d(hidden // 4, hidden // 4, 3, padding=1, groups=hidden // 4)
        self.dw5 = nn.Conv2d(hidden // 4, hidden // 4, 5, padding=2, groups=hidden // 4)
        self.dw7 = nn.Conv2d(hidden // 4, hidden // 4, 7, padding=3, groups=hidden // 4)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        # Multi-scale
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x2 = self.dw3(x2)
        x3 = self.dw5(x3)
        x4 = self.dw7(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv2(x)
        return x + residual


class TAUBlock(nn.Module):
    """Temporal Attention Unit (CVPR 2023)"""

    def __init__(self, dim):
        super().__init__()
        # 帧内空间注意力 (Intra-frame)
        self.spatial_norm = nn.GroupNorm(1, dim)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.Sigmoid(),
        )
        # 帧间时间注意力 (Inter-frame)
        self.temporal_norm = nn.GroupNorm(1, dim)
        self.temporal_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B*T, C, H, W]"""
        # Spatial
        s = self.spatial_attn(self.spatial_norm(x))
        x = x * s
        # Temporal (channel attention as temporal proxy)
        t = self.temporal_attn(self.temporal_norm(x))
        x = x * t.unsqueeze(-1).unsqueeze(-1)
        return x


class SimVPEncoder(nn.Module):
    """SimVP Spatial Encoder"""

    def __init__(self, in_channels, hid_S, N_S):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hid_S, 3, stride=2, padding=1),
            nn.GroupNorm(1, hid_S),
            nn.GELU(),
            *[InceptionBlock(hid_S) for _ in range(N_S)],
        )

    def forward(self, x):
        """x: [B*T, C, H, W] → [B*T, hid_S, H/2, W/2]"""
        return self.enc(x)


class SimVPDecoder(nn.Module):
    """SimVP Spatial Decoder"""

    def __init__(self, hid_S, out_channels, N_S):
        super().__init__()
        self.dec = nn.Sequential(
            *[InceptionBlock(hid_S) for _ in range(N_S)],
            nn.ConvTranspose2d(hid_S, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(x)


class SimVPTranslator(nn.Module):
    """SimVP Temporal Translator"""

    def __init__(self, in_seq, out_seq, hid_T, N_T, use_tau=False):
        super().__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq

        self.proj_in = nn.Conv2d(in_seq, hid_T, 1)
        if use_tau:
            self.blocks = nn.ModuleList([TAUBlock(hid_T) for _ in range(N_T)])
        else:
            self.blocks = nn.ModuleList([InceptionBlock(hid_T) for _ in range(N_T)])
        self.proj_out = nn.Conv2d(hid_T, out_seq, 1)

    def forward(self, x):
        """x: [B, T_in, C, H, W] → [B, T_out, C, H, W]"""
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b c) t h w')
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(x)
        x = rearrange(x, '(b c) t h w -> b t c h w', b=B)
        return x


class SimVP(nn.Module):
    """
    SimVP: Simpler yet Better Video Prediction
    支持 TAU (Temporal Attention Unit) 变体
    """

    def __init__(self, in_channels=3, hid_S=64, hid_T=256,
                 N_S=4, N_T=8, in_seq_len=10, out_seq_len=10,
                 img_size=128, use_tau=False):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        self.encoder = SimVPEncoder(in_channels, hid_S, N_S)
        self.translator = SimVPTranslator(in_seq_len, out_seq_len, hid_T, N_T, use_tau)
        self.decoder = SimVPDecoder(hid_S, in_channels, N_S)

    def forward(self, input_frames):
        """
        Args:
            input_frames: [B, T_in, C, H, W]
        Returns:
            pred_frames: [B, T_out, C, H, W]
        """
        B, T, C, H, W = input_frames.shape

        # Spatial encode each frame
        x = rearrange(input_frames, 'b t c h w -> (b t) c h w')
        x = self.encoder(x)  # [(B*T), hid_S, H/2, W/2]
        _, C_h, H_h, W_h = x.shape
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)

        # Temporal translation
        x = self.translator(x)  # [B, T_out, hid_S, H/2, W/2]

        # Spatial decode
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.decoder(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=self.out_seq_len)

        return x, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
