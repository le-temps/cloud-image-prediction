"""
CloudMambaNet - 方案A完整网络
SimVP 框架 + Mamba-LSTM 时间编码器 + DCN 空间解码器 + 双头输出
"""
import torch
import torch.nn as nn
from einops import rearrange

from .mamba_cell import MambaLSTMCell
from .dcn_decoder import DCNDecoder
from ..modules.patch_embed import PatchEmbed2D


class MambaEncoder(nn.Module):
    """Mamba-LSTM 时间编码器：逐帧处理输入序列"""

    def __init__(self, d_model=256, num_layers=4, d_state=16):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.cells = nn.ModuleList([
            MambaLSTMCell(d_model, d_state) for _ in range(num_layers)
        ])

    def forward(self, x_seq):
        """
        Args:
            x_seq: [B, T, D, H, W] patch embedded 输入序列
        Returns:
            h_list: 每层最终隐藏状态 list of [B, D, H, W]
            c_list: 每层最终细胞状态
            skip_features: encoder 中间特征 (用于 skip connection)
        """
        B, T, D, H, W = x_seq.shape
        device = x_seq.device

        # 初始化各层隐藏状态
        h_list = [cell.init_hidden(B, H, W, device)[0] for cell in self.cells]
        c_list = [cell.init_hidden(B, H, W, device)[1] for cell in self.cells]
        skip_features = []

        # 逐时间步前向传播
        for t in range(T):
            x_t = x_seq[:, t]  # [B, D, H, W]

            for l, cell in enumerate(self.cells):
                h_list[l], c_list[l] = cell(x_t, h_list[l], c_list[l])
                x_t = h_list[l]  # 下一层的输入是当前层的输出

            skip_features.append(x_t)

        return h_list, c_list, skip_features


class MambaDecoder(nn.Module):
    """Mamba-LSTM 时间解码器：自回归生成未来序列"""

    def __init__(self, d_model=256, num_layers=4, d_state=16):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.cells = nn.ModuleList([
            MambaLSTMCell(d_model, d_state) for _ in range(num_layers)
        ])

    def forward(self, h_list, c_list, out_seq_len):
        """
        Args:
            h_list: encoder 最终隐藏状态
            c_list: encoder 最终细胞状态
            out_seq_len: 预测序列长度
        Returns:
            outputs: [B, T_out, D, H, W] 预测特征序列
        """
        outputs = []
        x_t = h_list[-1]  # 用 encoder 最后一层的输出作为初始输入

        for t in range(out_seq_len):
            for l, cell in enumerate(self.cells):
                h_list[l], c_list[l] = cell(x_t, h_list[l], c_list[l])
                x_t = h_list[l]

            outputs.append(x_t)

        return torch.stack(outputs, dim=1)  # [B, T_out, D, H, W]


class CloudMambaNet(nn.Module):
    """
    完整的 CloudMambaNet
    Pipeline:
        输入 [B,T,C,H,W] → PatchEmbed → MambaEncoder → MambaDecoder
        → DCNDecoder → 双头输出 (云图重建 + 云mask分割)
    """

    def __init__(self, in_channels=3, img_size=128, patch_size=4,
                 d_model=256, num_layers=4, d_state=16,
                 deform_groups=4, out_seq_len=10, dual_head=True):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.out_seq_len = out_seq_len
        self.dual_head = dual_head

        # Patch Embedding
        self.patch_embed = PatchEmbed2D(in_channels, d_model, patch_size)
        self.patch_h = img_size // patch_size
        self.patch_w = img_size // patch_size

        # Mamba-LSTM Encoder & Decoder
        self.encoder = MambaEncoder(d_model, num_layers, d_state)
        self.decoder = MambaDecoder(d_model, num_layers, d_state)

        # DCN Spatial Decoder - 图像重建头
        self.spatial_decoder = DCNDecoder(
            in_dim=d_model, out_channels=in_channels,
            num_layers=num_layers, deform_groups=deform_groups,
            patch_size=patch_size,
        )

        # 云 Mask 分割头 (optional)
        if dual_head:
            self.mask_decoder = DCNDecoder(
                in_dim=d_model, out_channels=1,
                num_layers=num_layers, deform_groups=deform_groups,
                patch_size=patch_size,
            )

    def forward(self, input_frames):
        """
        Args:
            input_frames: [B, T_in, C, H, W]
        Returns:
            pred_frames: [B, T_out, C, H, W] 预测图像
            pred_masks: [B, T_out, 1, H, W] 预测云mask logits (if dual_head)
        """
        B, T_in, C, H, W = input_frames.shape

        # Patch Embedding for each frame
        x_seq = []
        for t in range(T_in):
            x_t = self.patch_embed(input_frames[:, t])  # [B, D, Hp, Wp]
            x_seq.append(x_t)
        x_seq = torch.stack(x_seq, dim=1)  # [B, T_in, D, Hp, Wp]

        # Encode
        h_list, c_list, skip_features = self.encoder(x_seq)

        # Decode temporal
        pred_features = self.decoder(h_list, c_list, self.out_seq_len)
        # [B, T_out, D, Hp, Wp]

        # Spatial decode each frame
        pred_frames = []
        pred_masks = []
        for t in range(self.out_seq_len):
            feat = pred_features[:, t]  # [B, D, Hp, Wp]
            frame = self.spatial_decoder(feat)
            frame = torch.sigmoid(frame)  # [0, 1]
            pred_frames.append(frame)

            if self.dual_head:
                mask = self.mask_decoder(feat)  # logits
                pred_masks.append(mask)

        pred_frames = torch.stack(pred_frames, dim=1)  # [B, T_out, C, H, W]

        if self.dual_head:
            pred_masks = torch.stack(pred_masks, dim=1)  # [B, T_out, 1, H, W]
            return pred_frames, pred_masks
        return pred_frames, None

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
