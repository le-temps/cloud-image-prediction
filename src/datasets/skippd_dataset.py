"""
SKIPP'D 数据集加载器
支持 Raw 模式: 加载 2048x2048 原始 jpg 图像，resize 到 img_size

数据预处理参考:
  - Nie et al., Solar Energy 2023 (SKIPP'D dataset paper)
  - Nie et al., Advances in Applied Energy 2024 (SkyGPT)
  - Nie et al., J. Renewable Sustainable Energy 2020 (Cloud detection: NRBR+CSL)

关键处理:
  1. 夜间帧过滤: 只保留 6:00-20:00 PST (相机工作时间)
  2. 鱼眼有效区域 mask: 只在圆形天空区域内计算云检测
  3. 云检测: NRBR 阈值法, (R-B)/(R+B) > threshold → 云
  4. 重复帧过滤: 检测 OpenCV 采集异常导致的重复图像
"""
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SKIPPDDataset(Dataset):
    """
    SKIPP'D 数据集：天空图像 + PV 功率
    按时间序列组织，每个样本为连续 (in_seq_len + out_seq_len) 帧

    目录结构:
        data/skippd_raw/
        ├── images/               # 从 raw tar 解压的 2048x2048 jpg 图像
        │   ├── 01/               # 月份目录
        │   │   ├── 24/           # 日期目录
        │   │   │   ├── 20190124060000.jpg
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        └── pv/
            └── 2019_pv_raw.csv

    数据来源:
        Stanford University, 鱼眼相机 Hikvision DS-2CD6362F-IV
        原始分辨率: 2048x2048, 采样频率: 1min
        时间范围: 2017.03 - 2019.12
        Raw: https://purl.stanford.edu/sm043zf7254
    """

    # 文件名时间戳格式: 20190124120010.jpg
    FILENAME_RE = re.compile(r'(\d{14})')
    DAYTIME_START_HOUR = 6   # 6:00 AM PST
    DAYTIME_END_HOUR = 20    # 8:00 PM PST

    def __init__(self, root, split='train', img_size=256,
                 in_seq_len=10, out_seq_len=10, temporal_stride=1,
                 transform=None, cloud_threshold=0.05):
        """
        Args:
            root: 数据集根目录
            split: 'train' / 'val' / 'test'
            img_size: 图像 resize 目标大小
            in_seq_len: 输入序列长度
            out_seq_len: 预测序列长度
            temporal_stride: 时间步间隔 (1=每分钟)
            transform: 额外的数据增强
            cloud_threshold: 云mask阈值 (R-B ratio), 参考作者论文取 0.05
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

        # 预计算鱼眼圆形 mask (resize 后的尺寸)
        self.fisheye_mask = self._build_fisheye_mask(img_size)

        # 加载图像路径列表 (已过滤夜间帧和重复帧)
        self.image_paths, self.timestamps = self._load_image_paths()

        # 构建有效序列索引 (检查时间连续性)
        self.sequences = self._build_sequences()

        # 图像变换
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # [0, 255] -> [0, 1]
        ])
        self.augment = transform

        print(f"[SKIPPDDataset] split={split}, images={len(self.image_paths)}, "
              f"sequences={len(self.sequences)}, img_size={img_size}")

    @staticmethod
    def _build_fisheye_mask(size):
        """
        构建鱼眼相机圆形有效区域 mask
        原始 2048x2048 图像中天空区域约为中心圆形 (半径~930px)
        resize 后按比例缩放
        """
        y, x = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing='ij',
        )
        # 鱼眼图像中心和半径 (原始 2048 中约 center=1024, radius=930)
        # resize 后等比例缩放
        center = size / 2.0
        radius = size * 930.0 / 2048.0  # ≈ 0.454 * size
        dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        mask = (dist <= radius).float()  # [H, W]
        return mask

    def _parse_timestamp(self, path):
        """从文件名解析时间戳: 20190124120010.jpg → datetime"""
        match = self.FILENAME_RE.search(path.stem)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
            except ValueError:
                return None
        return None

    def _load_image_paths(self):
        """加载排序后的图像路径, 过滤夜间帧和重复帧"""
        img_dir = self.root / 'images'
        if not img_dir.exists():
            img_dir = self.root

        # 递归搜索子目录
        extensions = ['*.jpg', '*.jpeg', '*.png']
        all_paths = []
        for ext in extensions:
            all_paths.extend(img_dir.rglob(ext))
        all_paths.sort()

        if len(all_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # 过滤: 解析时间戳 + 保留白天帧
        filtered_paths = []
        filtered_times = []
        prev_path_size = None

        for p in all_paths:
            ts = self._parse_timestamp(p)
            if ts is None:
                continue

            # 1. 夜间过滤: 只保留 6:00-20:00
            if ts.hour < self.DAYTIME_START_HOUR or ts.hour >= self.DAYTIME_END_HOUR:
                continue

            # 2. 重复帧过滤: 文件大小完全相同的连续帧很可能是 OpenCV 采集异常
            cur_size = p.stat().st_size
            if prev_path_size is not None and cur_size == prev_path_size:
                continue
            prev_path_size = cur_size

            filtered_paths.append(p)
            filtered_times.append(ts)

        n_filtered = len(all_paths) - len(filtered_paths)
        if n_filtered > 0:
            print(f"[SKIPPDDataset] Filtered {n_filtered} frames "
                  f"(night/duplicate), {len(filtered_paths)} remaining")

        if len(filtered_paths) == 0:
            raise FileNotFoundError(
                f"No valid daytime images found in {img_dir} "
                f"(checked {len(all_paths)} files)")

        return filtered_paths, filtered_times

    def _build_sequences(self):
        """
        构建有效的序列起始索引
        额外检查: 序列内帧的时间间隔应大致连续 (允许 ±30s 误差)
        按时间顺序划分 train/val/test (7:1:2)
        """
        total_images = len(self.image_paths)
        stride = self.temporal_stride
        required_len = self.total_len * stride
        max_gap_seconds = 90  # 允许的最大帧间隔 (正常1min, 允许1.5min)

        valid_starts = []
        for start in range(0, total_images - required_len + 1):
            # 检查序列内时间连续性
            is_valid = True
            for i in range(1, self.total_len):
                idx_cur = start + i * stride
                idx_prev = start + (i - 1) * stride
                dt = (self.timestamps[idx_cur] - self.timestamps[idx_prev]).total_seconds()
                if dt <= 0 or dt > max_gap_seconds:
                    is_valid = False
                    break
            if is_valid:
                valid_starts.append(start)

        # 按比例划分
        n_total = len(valid_starts)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.1)

        if self.split == 'train':
            return valid_starts[:n_train]
        elif self.split == 'val':
            return valid_starts[n_train:n_train + n_val]
        else:  # test
            return valid_starts[n_train + n_val:]

    def _compute_cloud_mask(self, img_tensor):
        """
        基于 NRBR (Normalized Red-Blue Ratio) 计算云 mask
        参考: Nie et al., J. Renewable Sustainable Energy 2020

        NRBR = (B - R) / (B + R), 云区域 NRBR 低 (≤ 0.05)
        等价于: (R - B) / (R + B) > threshold → 云

        额外应用鱼眼圆形 mask, 圆外区域不参与云检测

        Args:
            img_tensor: [C, H, W] 范围 [0, 1]
        Returns:
            mask: [1, H, W] 二值 mask (1=云, 0=非云/圆外)
        """
        r = img_tensor[0]  # Red channel
        b = img_tensor[2]  # Blue channel
        ratio = (r - b) / (r + b + 1e-8)
        cloud = (ratio > self.cloud_threshold).float()
        # 只在鱼眼有效区域内标记云
        cloud = cloud * self.fisheye_mask
        return cloud.unsqueeze(0)

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

        frames = torch.stack(frames)            # [T_total, C, H, W]
        cloud_masks = torch.stack(cloud_masks)  # [T_total, 1, H, W]

        # 分割输入和目标
        input_frames = frames[:self.in_seq_len]
        target_frames = frames[self.in_seq_len:]
        input_masks = cloud_masks[:self.in_seq_len]
        target_masks = cloud_masks[self.in_seq_len:]

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
