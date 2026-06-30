"""
ConvLSTM 基线模型
参考 Shi et al., NeurIPS 2015
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 4,
            kernel_size, padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    """多层 ConvLSTM 视频预测模型"""

    def __init__(self, in_channels=3, hidden_dims=[64, 64, 64, 64],
                 kernel_size=3, out_seq_len=10, img_size=128):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.out_seq_len = out_seq_len

        # 输入映射
        self.input_proj = nn.Conv2d(in_channels, hidden_dims[0], 1)

        # ConvLSTM 层
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            self.cells.append(ConvLSTMCell(in_ch, hidden_dims[i], kernel_size))

        # 输出映射
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_frames):
        """
        Args:
            input_frames: [B, T_in, C, H, W]
        Returns:
            pred_frames: [B, T_out, C, H, W]
            None
        """
        B, T, C, H, W = input_frames.shape
        device = input_frames.device

        # 初始化隐藏状态
        h_list = [torch.zeros(B, dim, H, W, device=device) for dim in self.hidden_dims]
        c_list = [torch.zeros(B, dim, H, W, device=device) for dim in self.hidden_dims]

        # Encode: 处理输入序列
        for t in range(T):
            x = self.input_proj(input_frames[:, t])
            for i, cell in enumerate(self.cells):
                h_list[i], c_list[i] = cell(x, h_list[i], c_list[i])
                x = h_list[i]

        # Decode: 自回归生成
        outputs = []
        x = h_list[-1]
        for t in range(self.out_seq_len):
            for i, cell in enumerate(self.cells):
                h_list[i], c_list[i] = cell(x, h_list[i], c_list[i])
                x = h_list[i]
            frame = self.output_proj(x)
            outputs.append(frame)
            x = self.input_proj(frame)

        return torch.stack(outputs, dim=1), None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
