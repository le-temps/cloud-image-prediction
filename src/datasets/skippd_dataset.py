"""SKIPP'D 数据集加载器"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SKIPPDDataset(Dataset):
    """
    SKIPP'D 数据集：天空图像 + PV 功率
    按时间序列组织，每个样本为连续 (in_seq_len + out_seq_len) 帧

    目录结构预期:
        data/skippd/
        ├── images/           # 所有图像按时间戳命名
        │   ├── 20200101_120000.jpg
        │   ├── 20200101_120100.jpg
        │   └── ...
        ├── metadata.csv      # 时间戳, PV功率, GHI 等
        └── splits/
            ├── train.txt     # 训练集序列索引
            ├── val.txt
            └── test.txt
    """

    def __init__(self, root, split='train', img_size=128,
                 in_seq_len=10, out_seq_len=10, temporal_stride=1,
                 transform=None, cloud_threshold=0.5):
        """
        Args:
            root: 数据集根目录
            split: 'train' / 'val' / 'test'
            img_size: 图像 resize 目标大小
            in_seq_len: 输入序列长度
            out_seq_len: 预测序列长度
            temporal_stride: 时间步间隔 (1=每分钟)
            transform: 额外的数据增强
            cloud_threshold: 云mask阈值 (基于 R-B ratio)
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.total_len = in_seq_len + out_seq_len
        self.temporal_stride = temporal_stride
        self.cloud_threshold = cloud_threshold

        # 加载图像路径列表
        self.image_paths = self._load_image_paths()

        # 构建有效序列索引
        self.sequences = self._build_sequences()

        # 图像变换
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # [0, 255] -> [0, 1]
        ])
        self.augment = transform

    def _load_image_paths(self):
        """加载排序后的图像路径"""
        img_dir = self.root / 'images'
        if not img_dir.exists():
            # 如果目录不存在，尝试直接读取根目录下的图像
            img_dir = self.root

        paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        return paths

    def _build_sequences(self):
        """
        构建有效的序列起始索引
        按时间顺序划分 train/val/test (7:1:2)
        """
        total_images = len(self.image_paths)
        seq_stride = self.temporal_stride
        required_len = self.total_len * seq_stride

        # 所有可能的起始索引
        all_starts = list(range(0, total_images - required_len + 1))

        # 按比例划分
        n_total = len(all_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        if self.split == 'train':
            sequences = all_starts[:n_train]
        elif self.split == 'val':
            sequences = all_starts[n_train:n_train + n_val]
        else:  # test
            sequences = all_starts[n_train + n_val:]

        return sequences

    def _compute_cloud_mask(self, img_tensor):
        """
        基于 Red-Blue Ratio 计算云mask
        云区域: R/B > threshold (云反射更多红光)
        Args:
            img_tensor: [C, H, W] 范围 [0, 1]
        Returns:
            mask: [1, H, W] 二值mask
        """
        r = img_tensor[0]  # Red channel
        b = img_tensor[2]  # Blue channel
        # R-B normalized difference
        ratio = (r - b) / (r + b + 1e-8)
        # 云区域 ratio > threshold
        mask = (ratio > self.cloud_threshold).float().unsqueeze(0)
        return mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start = self.sequences[idx]
        stride = self.temporal_stride

        # 加载连续帧
        frames = []
        cloud_masks = []
        for i in range(self.total_len):
            img_idx = start + i * stride
            img = Image.open(self.image_paths[img_idx]).convert('RGB')
            img_tensor = self.base_transform(img)
            frames.append(img_tensor)
            cloud_masks.append(self._compute_cloud_mask(img_tensor))

        frames = torch.stack(frames)        # [T_total, C, H, W]
        cloud_masks = torch.stack(cloud_masks)  # [T_total, 1, H, W]

        # 分割输入和目标
        input_frames = frames[:self.in_seq_len]           # [T_in, C, H, W]
        target_frames = frames[self.in_seq_len:]          # [T_out, C, H, W]
        input_masks = cloud_masks[:self.in_seq_len]       # [T_in, 1, H, W]
        target_masks = cloud_masks[self.in_seq_len:]      # [T_out, 1, H, W]

        # 数据增强 (训练时)
        if self.augment and self.split == 'train':
            input_frames, target_frames = self.augment(input_frames, target_frames)

        return {
            'input_frames': input_frames,       # [T_in, C, H, W]
            'target_frames': target_frames,     # [T_out, C, H, W]
            'input_masks': input_masks,         # [T_in, 1, H, W]
            'target_masks': target_masks,       # [T_out, 1, H, W]
        }


class TemporalAugmentation:
    """时序数据增强"""

    def __init__(self, flip_prob=0.5, rotate_prob=0.5):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, input_frames, target_frames):
        # 空间翻转 (对序列所有帧一致应用)
        if torch.rand(1).item() < self.flip_prob:
            input_frames = torch.flip(input_frames, [-1])  # 水平翻转
            target_frames = torch.flip(target_frames, [-1])

        if torch.rand(1).item() < self.flip_prob:
            input_frames = torch.flip(input_frames, [-2])  # 垂直翻转
            target_frames = torch.flip(target_frames, [-2])

        # 90度旋转
        if torch.rand(1).item() < self.rotate_prob:
            k = torch.randint(1, 4, (1,)).item()
            input_frames = torch.rot90(input_frames, k, [-2, -1])
            target_frames = torch.rot90(target_frames, k, [-2, -1])

        return input_frames, target_frames
