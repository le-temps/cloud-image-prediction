"""
PhyCell - 物理约束层
将 PDE 物理先验 (平流方程) 嵌入网络
参考 PhyDNet (CVPR 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhyCell_Cell(nn.Module):
    """
    物理约束单元: 用可学习的卷积核近似偏微分算子
    内嵌 ∂u/∂t + v·∇u = 0 的离散近似
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 物理约束核: 近似空间微分算子
        # 使用多个卷积核表示不同阶的偏导数
        self.F_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
        )

        # 输入门: 控制新信息流入
        self.gate_conv = nn.Conv2d(input_dim * 2, input_dim, kernel_size=3, padding=1)

    def forward(self, x, h_prev):
        """
        Args:
            x: [B, D, H, W] 当前输入
            h_prev: [B, D, H, W] 前一时刻隐藏状态
        Returns:
            h: [B, D, H, W] 当前隐藏状态 (物理约束后)
        """
        # 物理演化: h_prev + F(h_prev) ≈ h_prev + Δt * (physical dynamics)
        h_phys = h_prev + self.F_conv(h_prev)

        # 门控融合物理演化和观测输入
        gate = torch.sigmoid(self.gate_conv(torch.cat([x, h_phys], dim=1)))
        h = gate * x + (1 - gate) * h_phys

        return h

    def init_hidden(self, batch_size, height, width, device):
        return torch.zeros(batch_size, self.input_dim, height, width, device=device)


class PhyCell(nn.Module):
    """
    完整的物理约束模块
    分离物理动力学分支 + 残差补偿分支
    """

    def __init__(self, input_dim, hidden_dim=64, kernel_size=7, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        # 物理分支: PDE-constrained
        self.phy_cells = nn.ModuleList([
            PhyCell_Cell(input_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])

        # 残差分支: 学习物理模型无法捕捉的部分 (云生消等)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1),
        )

        # 融合层
        self.fusion = nn.Conv2d(input_dim * 2, input_dim, 1)

    def forward(self, x, h_list=None):
        """
        Args:
            x: [B, D, H, W] 输入特征
            h_list: list of hidden states
        Returns:
            out: [B, D, H, W]
            h_list: 更新后的隐藏状态
        """
        B, D, H, W = x.shape
        device = x.device

        if h_list is None:
            h_list = [cell.init_hidden(B, H, W, device) for cell in self.phy_cells]

        # 物理分支
        h_phy = x
        for i, cell in enumerate(self.phy_cells):
            h_list[i] = cell(h_phy, h_list[i])
            h_phy = h_list[i]

        # 残差分支
        h_res = self.residual_conv(x)

        # 融合
        out = self.fusion(torch.cat([h_phy, h_res], dim=1))

        return out, h_list
