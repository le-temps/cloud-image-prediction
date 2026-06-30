"""
CloudDiffNet - 方案C完整网络
云分割前置网络 + 分割引导条件扩散模型
"""
import torch
import torch.nn as nn

from .cloud_segmentor import CloudSegmentor
from .guided_diffusion import GuidedDiffusion


class CloudDiffNet(nn.Module):
    """
    CloudDiffNet 完整网络
    Pipeline:
        阶段1: 云分割前置网络提取 mask 序列
        阶段2: 条件扩散模型在 mask 引导下生成未来云图
        阶段3: 多次采样给出概率预测
    """

    def __init__(self, in_channels=3, img_size=128,
                 in_seq_len=10, out_seq_len=10,
                 channels=[64, 128, 256, 512], cond_channels=64,
                 timesteps=1000, beta_schedule='cosine',
                 inference_steps=50, num_samples=20,
                 freeze_segmentor=True):
        super().__init__()
        self.in_channels = in_channels
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.num_samples = num_samples
        self.freeze_segmentor = freeze_segmentor

        # 云分割前置网络
        self.segmentor = CloudSegmentor(in_channels)

        if freeze_segmentor:
            for param in self.segmentor.parameters():
                param.requires_grad = False

        # 条件扩散模型
        self.diffusion = GuidedDiffusion(
            in_channels=in_channels,
            img_size=img_size,
            num_frames=out_seq_len,
            channels=channels,
            cond_channels=cond_channels,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            inference_steps=inference_steps,
        )

    def forward(self, input_frames, target_frames=None):
        """
        Args:
            input_frames: [B, T_in, C, H, W]
            target_frames: [B, T_out, C, H, W] (训练时提供)
        Returns:
            训练: loss (scalar)
            推理: pred_frames [B, T_out, C, H, W], pred_masks [B, T_out, 1, H, W]
        """
        # 分割输入帧
        if self.freeze_segmentor:
            with torch.no_grad():
                input_masks = self.segmentor.segment_sequence(input_frames)
        else:
            input_masks = self.segmentor.segment_sequence(input_frames)

        input_masks = torch.sigmoid(input_masks)

        if self.training and target_frames is not None:
            # 训练: 返回扩散模型loss
            loss = self.diffusion(input_frames, input_masks, target_frames)
            return loss
        else:
            # 推理: 采样生成
            samples = self.diffusion.sample(
                input_frames, input_masks,
                num_samples=self.num_samples,
            )
            # samples: [N, B, T_out, C, H, W]

            # 均值预测
            pred_frames = samples.mean(dim=0)  # [B, T_out, C, H, W]

            # 分割预测帧获得 mask
            with torch.no_grad():
                pred_masks = self.segmentor.segment_sequence(pred_frames)

            # 也可以输出不确定性
            pred_std = samples.std(dim=0)  # [B, T_out, C, H, W]

            return pred_frames, pred_masks

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
