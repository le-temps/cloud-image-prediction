"""
PredRNN++ 基线模型
参考 Wang et al., NeurIPS 2017 / ICML 2018
"""
import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """PredRNN 的核心: Causal LSTM + Spatiotemporal Memory"""

    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # 标准 LSTM gates
        self.gates_h = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 4,
            kernel_size, padding=padding,
        )
        # Spatiotemporal memory gates
        self.gates_m = nn.Conv2d(
            in_channels + hidden_channels, hidden_channels * 4,
            kernel_size, padding=padding,
        )
        # Memory fusion
        self.fusion = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(self, x, h, c, m):
        # Temporal memory (standard LSTM)
        combined_h = torch.cat([x, h], dim=1)
        gates_h = self.gates_h(combined_h)
        i_h, f_h, o_h, g_h = gates_h.chunk(4, dim=1)

        i_h = torch.sigmoid(i_h)
        f_h = torch.sigmoid(f_h)
        o_h = torch.sigmoid(o_h)
        g_h = torch.tanh(g_h)
        c_new = f_h * c + i_h * g_h

        # Spatiotemporal memory
        combined_m = torch.cat([x, m], dim=1)
        gates_m = self.gates_m(combined_m)
        i_m, f_m, o_m, g_m = gates_m.chunk(4, dim=1)

        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        o_m = torch.sigmoid(o_m)
        g_m = torch.tanh(g_m)
        m_new = f_m * m + i_m * g_m

        # Fusion
        h_new = o_h * torch.tanh(self.fusion(torch.cat([c_new, m_new], dim=1)))

        return h_new, c_new, m_new


class PredRNN(nn.Module):
    """PredRNN++ 视频预测模型"""

    def __init__(self, in_channels=3, hidden_dims=[64, 64, 64, 64],
                 kernel_size=5, out_seq_len=10, img_size=128, version='v2'):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.out_seq_len = out_seq_len

        self.input_proj = nn.Conv2d(in_channels, hidden_dims[0], 1)

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            self.cells.append(
                SpatioTemporalLSTMCell(in_ch, hidden_dims[i], kernel_size)
            )

        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_frames):
        B, T, C, H, W = input_frames.shape
        device = input_frames.device

        h_list = [torch.zeros(B, dim, H, W, device=device) for dim in self.hidden_dims]
        c_list = [torch.zeros(B, dim, H, W, device=device) for dim in self.hidden_dims]
        m = torch.zeros(B, self.hidden_dims[0], H, W, device=device)

        # Encode
        for t in range(T):
            x = self.input_proj(input_frames[:, t])
            for i, cell in enumerate(self.cells):
                h_list[i], c_list[i], m = cell(x, h_list[i], c_list[i], m)
                x = h_list[i]

        # Decode
        outputs = []
        x = h_list[-1]
        for t in range(self.out_seq_len):
            for i, cell in enumerate(self.cells):
                h_list[i], c_list[i], m = cell(x, h_list[i], c_list[i], m)
                x = h_list[i]
            frame = self.output_proj(x)
            outputs.append(frame)
            x = self.input_proj(frame)

        return torch.stack(outputs, dim=1), None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
