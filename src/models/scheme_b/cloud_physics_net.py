"""
CloudPhysicsNet - 方案B完整网络
多尺度 Cuboid Attention + AFNO 频域空间混合 + PhyCell 物理约束
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .cuboid_attention import CuboidTransformerBlock
from .afno_block import AFNOBlock
from .phycell import PhyCell
from ..modules.patch_embed import PatchEmbed2D


class CloudPhysicsNet(nn.Module):
    """
    CloudPhysicsNet 完整网络
    Pipeline:
        输入 [B,T,C,H,W] → PatchEmbed → Cuboid Attention 时空编码
        → AFNO 频域空间混合 → PhyCell 物理约束 → 输出预测
    """

    def __init__(self, in_channels=3, img_size=128, patch_size=4,
                 d_model=256, num_heads=8, cuboid_size=(2, 4, 4),
                 num_cuboid_layers=4, afno_blocks=8, afno_hidden_ratio=1.0,
                 phycell_kernel=7, phycell_dim=64,
                 out_seq_len=10, in_seq_len=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_seq_len = out_seq_len
        self.in_seq_len = in_seq_len
        self.d_model = d_model
        self.patch_size = patch_size

        # Patch Embedding
        self.patch_embed = PatchEmbed2D(in_channels, d_model, patch_size)
        self.patch_h = img_size // patch_size
        self.patch_w = img_size // patch_size

        # 位置编码 (时空)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, in_seq_len, self.patch_h, self.patch_w, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Cuboid Transformer Encoder
        self.cuboid_layers = nn.ModuleList([
            CuboidTransformerBlock(d_model, num_heads, cuboid_size)
            for _ in range(num_cuboid_layers)
        ])

        # AFNO Block (频域空间混合)
        self.afno = AFNOBlock(d_model, afno_blocks, afno_hidden_ratio)

        # PhyCell (物理约束)
        self.phycell = PhyCell(d_model, phycell_dim, phycell_kernel)

        # 时间预测头: 从 T_in 帧映射到 T_out 帧
        self.temporal_proj = nn.Conv1d(in_seq_len, out_seq_len, kernel_size=1)

        # 空间恢复
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model // 2,
                               kernel_size=patch_size, stride=patch_size),
            nn.GroupNorm(1, d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input_frames):
        """
        Args:
            input_frames: [B, T_in, C, H, W]
        Returns:
            pred_frames: [B, T_out, C, H, W]
            None (无mask头，保持接口一致)
        """
        B, T, C, H, W = input_frames.shape

        # Patch Embedding for each frame
        x = []
        for t in range(T):
            x_t = self.patch_embed(input_frames[:, t])  # [B, D, Hp, Wp]
            x.append(x_t)
        x = torch.stack(x, dim=1)  # [B, T, D, Hp, Wp]

        # 转换为 [B, T, Hp, Wp, D] 用于 Cuboid Attention
        x = x.permute(0, 1, 3, 4, 2)

        # 加位置编码
        x = x + self.pos_embed[:, :T]

        # Cuboid Transformer
        for layer in self.cuboid_layers:
            x = layer(x)  # [B, T, Hp, Wp, D]

        # AFNO (在每帧上独立应用)
        B, T, Hp, Wp, D = x.shape
        x_flat = x.reshape(B * T, Hp, Wp, D)
        x_flat = self.afno(x_flat)
        x = x_flat.reshape(B, T, Hp, Wp, D)

        # PhyCell (逐帧处理，带时间状态传递)
        x = x.permute(0, 1, 4, 2, 3)  # [B, T, D, Hp, Wp]
        phy_outputs = []
        h_list = None
        for t in range(T):
            out, h_list = self.phycell(x[:, t], h_list)
            phy_outputs.append(out)
        x = torch.stack(phy_outputs, dim=1)  # [B, T, D, Hp, Wp]

        # 时间映射: T_in → T_out
        x = x.reshape(B, T, -1)  # [B, T, D*Hp*Wp]
        x = self.temporal_proj(x)  # [B, T_out, D*Hp*Wp]
        x = x.reshape(B, self.out_seq_len, self.d_model, Hp, Wp)

        # Spatial Decode each frame
        pred_frames = []
        for t in range(self.out_seq_len):
            frame = self.decoder(x[:, t])  # [B, C, H, W]
            pred_frames.append(frame)

        pred_frames = torch.stack(pred_frames, dim=1)  # [B, T_out, C, H, W]
        return pred_frames, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
