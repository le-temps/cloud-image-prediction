"""
统一训练入口
支持所有基线和方案的训练，通过配置文件切换
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from datasets import build_dataset
from models.baselines.convlstm import ConvLSTM
from models.baselines.predrnn import PredRNN
from models.baselines.simvp import SimVP
from models.scheme_a.cloud_mamba_net import CloudMambaNet
from models.scheme_b.cloud_physics_net import CloudPhysicsNet
from models.scheme_c.cloud_diff_net import CloudDiffNet
from models.modules.losses import CombinedLoss, PhysicsLoss
from models.modules.metrics import MetricCalculator
from utils.distributed import (setup_distributed, cleanup_distributed,
                                get_model_ddp, get_dataloader, is_main_process)
from utils.logger import setup_logger, ExperimentTracker, save_checkpoint


MODEL_REGISTRY = {
    'convlstm': ConvLSTM,
    'predrnn': PredRNN,
    'simvp': lambda **kw: SimVP(use_tau=False, **kw),
    'simvp_tau': lambda **kw: SimVP(use_tau=True, **kw),
    'scheme_a': CloudMambaNet,
    'scheme_b': CloudPhysicsNet,
    'scheme_c': CloudDiffNet,
}


def build_model(config):
    """构建模型"""
    model_name = config['model']['name']
    model_cfg = config['model'].copy()
    model_cfg.pop('name')
    model_cfg.pop('arch', None)

    # 通用参数
    dataset_cfg = config.get('dataset', {})
    model_cfg.setdefault('in_channels', dataset_cfg.get('channels', 3))
    model_cfg.setdefault('img_size', dataset_cfg.get('img_size', 128))
    model_cfg.setdefault('out_seq_len', dataset_cfg.get('out_seq_len', 10))

    if model_name in ['simvp', 'simvp_tau']:
        model_cfg.setdefault('in_seq_len', dataset_cfg.get('in_seq_len', 10))
        model_cfg.pop('spatio_kernel', None)
        model_cfg.pop('act_inplace', None)
    elif model_name == 'scheme_b':
        model_cfg.setdefault('in_seq_len', dataset_cfg.get('in_seq_len', 10))
        model_cfg.pop('multiscale_patches', None)
    elif model_name == 'scheme_c':
        model_cfg.setdefault('in_seq_len', dataset_cfg.get('in_seq_len', 10))
        model_cfg.pop('segmentor', None)
        model_cfg.pop('diffusion', None)
    elif model_name in ['convlstm', 'predrnn']:
        model_cfg.pop('version', None)

    # 清理不适用于模型构造的参数
    for key in ['dual_head']:
        model_cfg.pop(key, None)

    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(**model_cfg)


def build_optimizer(model, config):
    """构建优化器"""
    opt_cfg = config['training']['optimizer']
    if opt_cfg['name'] == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg.get('weight_decay', 1e-4),
        )
    elif opt_cfg['name'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=opt_cfg['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")


def build_scheduler(optimizer, config):
    """构建学习率调度器"""
    sch_cfg = config['training']['scheduler']
    if sch_cfg['name'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg['T_max'],
            eta_min=sch_cfg.get('eta_min', 1e-6),
        )
    else:
        return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, config,
                    epoch, logger, tracker, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    model_name = config['model']['name']

    pbar = tqdm(loader, desc=f'Epoch {epoch}', disable=not is_main_process())
    for batch in pbar:
        input_frames = batch['input_frames'].to(device)
        target_frames = batch['target_frames'].to(device)
        target_masks = batch['target_masks'].to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            if model_name == 'scheme_c':
                # 方案C: 扩散模型直接返回loss
                loss = model(input_frames, target_frames)
                losses = {'total': loss, 'diffusion': loss}
            else:
                pred_frames, pred_masks = model(input_frames)
                loss, losses = criterion(
                    pred_frames, target_frames,
                    pred_masks, target_masks,
                )

        scaler.scale(loss).backward()

        # Gradient clipping
        if config['training'].get('grad_clip', 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config['training']['grad_clip']
            )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    if is_main_process():
        tracker.log({'train/loss': avg_loss, 'train/lr': optimizer.param_groups[0]['lr']},
                    step=epoch)
    return avg_loss


@torch.no_grad()
def validate(model, loader, metric_calc, config, epoch, tracker, device):
    """验证"""
    model.eval()
    model_name = config['model']['name']
    all_metrics = []

    for batch in loader:
        input_frames = batch['input_frames'].to(device)
        target_frames = batch['target_frames'].to(device)
        target_masks = batch['target_masks'].to(device)

        if model_name == 'scheme_c':
            pred_frames, pred_masks = model(input_frames)
        else:
            pred_frames, pred_masks = model(input_frames)

        metrics = metric_calc.compute_all(
            pred_frames, target_frames,
            pred_masks if pred_masks is not None else None,
            target_masks,
        )
        all_metrics.append(metrics)

    # 平均指标
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    if is_main_process():
        tracker.log({f'val/{k}': v for k, v in avg_metrics.items()}, step=epoch)

    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/skippd.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--resume', type=str, default=None, help='检查点路径')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    config['dataset'] = dataset_config['dataset']

    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # 设置随机种子
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 输出目录
    model_name = config['model']['name']
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger & Tracker
    logger = setup_logger(model_name, str(output_dir), rank)
    tracker = ExperimentTracker(
        config, str(output_dir),
        use_wandb=not args.no_wandb and is_main_process(),
    )

    # 构建数据集
    train_dataset = build_dataset(config['dataset'], split='train')
    val_dataset = build_dataset(config['dataset'], split='val')

    batch_size = config['training']['batch_size']
    train_loader, train_sampler = get_dataloader(train_dataset, batch_size, is_train=True)
    val_loader, _ = get_dataloader(val_dataset, batch_size, is_train=False)

    # 构建模型
    model = build_model(config)
    if is_main_process():
        logger.info(f"Model: {model_name}, Params: {model.get_num_params() / 1e6:.2f}M")
    model = get_model_ddp(model, local_rank)

    # 优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler()

    # 损失函数
    loss_cfg = config.get('loss', {})
    criterion = CombinedLoss(
        mse_weight=loss_cfg.get('mse_weight', 1.0),
        ssim_weight=loss_cfg.get('ssim_weight', 0.5),
        lpips_weight=loss_cfg.get('lpips_weight', 0.1),
        bce_weight=loss_cfg.get('bce_weight', 0.0),
    )

    # 评估器
    metric_calc = MetricCalculator(device)

    # 训练循环
    epochs = config['training']['epochs']
    best_ssim = 0
    start_epoch = 0

    if args.resume:
        from utils.logger import load_checkpoint
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, epoch, logger, tracker, device,
        )

        if scheduler:
            scheduler.step()

        # 验证 (每 5 epoch)
        if epoch % 5 == 0 or epoch == epochs - 1:
            metrics = validate(model, val_loader, metric_calc, config, epoch, tracker, device)

            if is_main_process():
                logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, "
                            f"SSIM={metrics.get('ssim', 0):.4f}, "
                            f"PSNR={metrics.get('psnr', 0):.2f}")

                # 保存最佳模型
                if metrics.get('ssim', 0) > best_ssim:
                    best_ssim = metrics['ssim']
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, metrics,
                        str(output_dir / 'best.pth'),
                    )

        # 定期保存
        if is_main_process() and epoch % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, {},
                str(output_dir / f'epoch_{epoch}.pth'),
            )

    # 保存最终模型
    if is_main_process():
        save_checkpoint(
            model, optimizer, scheduler, epochs - 1, {},
            str(output_dir / 'last.pth'),
        )
        tracker.finish()
        logger.info("Training complete!")

    cleanup_distributed()


if __name__ == '__main__':
    main()
