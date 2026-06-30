"""
NREL TSI-880 数据集加载器
来源: NREL Solar Radiation Research Laboratory (SRRL)
设备: TSI-880 全天空成像仪, 原始分辨率 1024x1024
用途: 跨域泛化验证

目录结构:
    data/nrel_tsi/
    ├── images/           # TSI-880 拍摄的全天空图像
    │   ├── 20180101_0800.jpg
    │   ├── 20180101_0801.jpg
    │   └── ...
    └── irradiance.csv    # GHI/DNI 辐照度数据 (optional)

数据获取: https://midcdmz.nrel.gov/
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class NRELTSIDataset(Dataset):
    """
    NREL TSI-880 全天空成像仪数据集
    原始分辨率 1024x1024，resize 到 img_size
    主要用于跨域泛化验证 (不同站点、不同气候)
    """

    def __init__(self, root, split='test', img_size=256,
                 in_seq_len=10, out_seq_len=10, temporal_stride=1,
                 cloud_threshold=0.3):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.total_len = in_seq_len + out_seq_len
        self.temporal_stride = temporal_stride
        self.cloud_threshold = cloud_threshold

        self.image_paths = self._load_image_paths()
        self.sequences = self._build_sequences()

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def _load_image_paths(self):
        """递归搜索所有图像并按文件名排序 (时间顺序)"""
        img_dir = self.root / 'images'
        if not img_dir.exists():
            img_dir = self.root

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        paths = []
        for ext in extensions:
            paths.extend(sorted(img_dir.rglob(ext)))
        paths.sort()

        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        return paths

    def _build_sequences(self):
        """按时间顺序划分 train/val/test (7:1:2)"""
        total_images = len(self.image_paths)
        required_len = self.total_len * self.temporal_stride
        all_starts = list(range(0, total_images - required_len + 1))

        n_total = len(all_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        if self.split == 'train':
            return all_starts[:n_train]
        elif self.split == 'val':
            return all_starts[n_train:n_train + n_val]
        else:
            return all_starts[n_train + n_val:]

    def _compute_cloud_mask(self, img_tensor):
        """R-B ratio 云检测"""
        r = img_tensor[0]
        b = img_tensor[2]
        ratio = (r - b) / (r + b + 1e-8)
        mask = (ratio > self.cloud_threshold).float().unsqueeze(0)
        return mask

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start = self.sequences[idx]
        stride = self.temporal_stride

        frames = []
        cloud_masks = []
        for i in range(self.total_len):
            img_idx = start + i * stride
            img = Image.open(self.image_paths[img_idx]).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)
            cloud_masks.append(self._compute_cloud_mask(img_tensor))

        frames = torch.stack(frames)
        cloud_masks = torch.stack(cloud_masks)

        input_frames = frames[:self.in_seq_len]
        target_frames = frames[self.in_seq_len:]
        input_masks = cloud_masks[:self.in_seq_len]
        target_masks = cloud_masks[self.in_seq_len:]

        return {
            'input_frames': input_frames,
            'target_frames': target_frames,
            'input_masks': input_masks,
            'target_masks': target_masks,
        }
