"""数据集工厂函数"""
from .skippd_dataset import SKIPPDDataset, TemporalAugmentation
from .nrel_dataset import NRELTSIDataset


def build_dataset(config, split='train'):
    """
    根据配置构建数据集
    Args:
        config: dataset 配置字典
        split: 'train' / 'val' / 'test'
    """
    dataset_name = config['name'].lower()
    common_args = {
        'img_size': config.get('img_size', 256),
        'in_seq_len': config.get('in_seq_len', 10),
        'out_seq_len': config.get('out_seq_len', 10),
        'temporal_stride': config.get('temporal_stride', 1),
        'cloud_threshold': config.get('cloud_threshold', 0.3),
    }

    augment = TemporalAugmentation() if split == 'train' else None

    if dataset_name in ['skippd', 'skippd_raw']:
        return SKIPPDDataset(
            root=config['root'],
            split=split,
            transform=augment,
            **common_args,
        )
    elif dataset_name == 'nrel_tsi':
        return NRELTSIDataset(
            root=config['root'],
            split=split,
            **common_args,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
