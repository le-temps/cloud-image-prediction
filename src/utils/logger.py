"""日志和实验跟踪工具"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def setup_logger(name, log_dir, rank=0):
    """创建 logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # Formatter
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


class ExperimentTracker:
    """实验跟踪器，支持 wandb + 本地 JSON"""

    def __init__(self, config, log_dir, use_wandb=True, project_name='cloud-prediction'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.use_wandb = use_wandb and HAS_WANDB

        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=config.get('model', {}).get('name', 'unknown'),
                config=config,
                dir=str(self.log_dir),
            )

    def log(self, metrics, step=None, prefix=''):
        """记录指标"""
        if prefix:
            metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}

        self.metrics_history.append({'step': step, **metrics})

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_images(self, images_dict, step=None):
        """记录图像 (用于可视化)"""
        if self.use_wandb:
            wandb_images = {
                k: wandb.Image(v) for k, v in images_dict.items()
            }
            wandb.log(wandb_images, step=step)

    def save_metrics(self):
        """保存指标到本地"""
        path = self.log_dir / 'metrics.json'
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def finish(self):
        """结束跟踪"""
        self.save_metrics()
        if self.use_wandb:
            wandb.finish()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """保存训练检查点"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    ckpt = torch.load(path, map_location='cpu')
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(ckpt['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    return ckpt['epoch'], ckpt.get('metrics', {})
