"""
Vision Mamba Cell - 时间建模单元
将 Vision Mamba (SSM) 与 LSTM 门控机制结合，用于时空预测
参考 VMRNN (arXiv:2403.16536) 的核心思路，针对云图预测进行适配
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class SS2D(nn.Module):
    """
    2D Selective Scan - 将2D特征图沿4条路径展开为1D序列后送入 Mamba
    4条扫描路径: 左→右, 右→左, 上→下, 下→上
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback: 简化版用 1D Conv + Gating 近似
            self.proj = nn.Linear(d_model, d_model * 2)
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv,
                                  padding=d_conv - 1, groups=d_model)
            self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: [B, D, H, W]
        Returns:
            [B, D, H, W]
        """
        B, D, H, W = x.shape

        # 4方向扫描
        # 路径1: 逐行从左到右
        x_lr = rearrange(x, 'b d h w -> b (h w) d')
        # 路径2: 逐行从右到左
        x_rl = torch.flip(x_lr, [1])
        # 路径3: 逐列从上到下
        x_td = rearrange(x, 'b d h w -> b (w h) d')
        # 路径4: 逐列从下到上
        x_bu = torch.flip(x_td, [1])

        # 通过 Mamba/SSM 处理
        if HAS_MAMBA:
            y_lr = self.mamba(x_lr)
            y_rl = torch.flip(self.mamba(x_rl), [1])
            y_td = self.mamba(x_td)
            y_bu = torch.flip(self.mamba(x_bu), [1])
        else:
            y_lr = self._fallback_ssm(x_lr)
            y_rl = torch.flip(self._fallback_ssm(x_rl), [1])
            y_td = self._fallback_ssm(x_td)
            y_bu = torch.flip(self._fallback_ssm(x_bu), [1])

        # 聚合4条路径
        y_spatial = rearrange(y_lr + y_rl, 'b (h w) d -> b d h w', h=H, w=W)
        y_channel = rearrange(y_td + y_bu, 'b (w h) d -> b d h w', h=H, w=W)
        out = (y_spatial + y_channel) / 4.0

        return out

    def _fallback_ssm(self, x):
        """Fallback 实现 (无 mamba-ssm 包时)"""
        B, L, D = x.shape
        xz = self.proj(x)
        x_gate, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_gate.transpose(1, 2))[:, :, :L].transpose(1, 2)
        y = self.out_proj(F.silu(x_conv) * F.silu(z))
        return y


class VSSBlock(nn.Module):
    """Vision State Space Block - 结合 SS2D + LayerNorm + FFN"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, d_model)
        self.ss2d = SS2D(d_model, d_state, d_conv, expand)
        self.norm2 = nn.GroupNorm(1, d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_model, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, d_model, 1),
        )

    def forward(self, x):
        """x: [B, D, H, W]"""
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaLSTMCell(nn.Module):
    """
    Mamba-LSTM Cell: 结合 Vision Mamba 的空间建模与 LSTM 的时间门控
    - VSS Block 负责空间特征提取 (线性复杂度全局建模)
    - LSTM gate 负责时间信息传递 (记忆与遗忘)
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model

        # 空间建模: Vision Mamba
        self.vss = VSSBlock(d_model, d_state)

        # 时间门控: LSTM-style gates
        # input, forget, output, cell gates
        self.gates = nn.Conv2d(d_model * 2, d_model * 4, kernel_size=3, padding=1)
        self.norm_h = nn.GroupNorm(1, d_model)
        self.norm_c = nn.GroupNorm(1, d_model)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: [B, D, H, W] 当前帧特征
            h_prev: [B, D, H, W] 前一时刻隐藏状态
            c_prev: [B, D, H, W] 前一时刻细胞状态
        Returns:
            h: [B, D, H, W] 当前隐藏状态
            c: [B, D, H, W] 当前细胞状态
        """
        # 空间特征提取 (Mamba)
        x_spatial = self.vss(x)

        # LSTM 门控计算
        combined = torch.cat([x_spatial, self.norm_h(h_prev)], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(self.norm_c(c))

        return h, c

    def init_hidden(self, batch_size, height, width, device):
        """初始化隐藏状态"""
        h = torch.zeros(batch_size, self.d_model, height, width, device=device)
        c = torch.zeros(batch_size, self.d_model, height, width, device=device)
        return h, c
