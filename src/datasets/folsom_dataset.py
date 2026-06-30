"""Folsom 数据集加载器 (TSI-880 全天空成像仪)"""
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class FolsomDataset(Dataset):
    """
    Folsom 数据集：TSI-880 全天空成像仪
    主要用于跨域泛化验证

    目录结构预期:
        data/folsom/
        ├── images/
        │   ├── 20180101_080000.jpg
        │   └── ...
        └── irradiance.csv    # GHI数据 (optional)
    """

    def __init__(self, root, split='test', img_size=128,
                 in_seq_len=10, out_seq_len=10, temporal_stride=1,
                 cloud_threshold=0.5):
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
        img_dir = self.root / 'images'
        if not img_dir.exists():
            img_dir = self.root
        paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        return paths

    def _build_sequences(self):
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
