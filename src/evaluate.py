"""
统一评估入口
加载训练好的模型，在指定数据集上评估所有指标
"""
import argparse
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

from datasets import build_dataset
from train import build_model
from models.modules.metrics import MetricCalculator
from utils.distributed import get_dataloader
from utils.visualization import visualize_prediction, plot_metrics_over_time
from utils.logger import load_checkpoint


@torch.no_grad()
def evaluate(model, loader, metric_calc, device, output_dir, num_vis=10):
    """
    全量评估
    Returns:
        avg_metrics: 平均指标
        frame_metrics: 逐帧指标
    """
    model.eval()
    all_metrics = []
    all_frame_metrics = []
    vis_count = 0

    for batch in tqdm(loader, desc='Evaluating'):
        input_frames = batch['input_frames'].to(device)
        target_frames = batch['target_frames'].to(device)
        target_masks = batch['target_masks'].to(device)

        # 前向推理
        pred_frames, pred_masks = model(input_frames)

        # 全局指标
        metrics = metric_calc.compute_all(
            pred_frames, target_frames,
            pred_masks if pred_masks is not None else None,
            target_masks,
        )
        all_metrics.append(metrics)

        # 逐帧指标
        frame_m = metric_calc.compute_per_frame(pred_frames, target_frames)
        all_frame_metrics.append(frame_m)

        # Smart Persistence baseline (用最后一帧重复)
        persistence = input_frames[:, -1:].expand_as(target_frames)
        metrics['forecast_skill'] = metric_calc.forecast_skill(
            pred_frames, target_frames, persistence
        )

        # 可视化前 N 个样本
        if vis_count < num_vis:
            for b in range(min(input_frames.shape[0], num_vis - vis_count)):
                vis_path = output_dir / 'vis' / f'sample_{vis_count:04d}.png'
                visualize_prediction(
                    input_frames[b].cpu(),
                    pred_frames[b].cpu(),
                    target_frames[b].cpu(),
                    str(vis_path),
                    cloud_mask_pred=pred_masks[b].cpu() if pred_masks is not None else None,
                    cloud_mask_target=target_masks[b].cpu(),
                )
                vis_count += 1
                if vis_count >= num_vis:
                    break

    # 计算平均
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    # 计算平均逐帧指标
    avg_frame_metrics = {}
    for key in all_frame_metrics[0]:
        T = len(all_frame_metrics[0][key])
        avg_frame_metrics[key] = [
            sum(m[key][t] for m in all_frame_metrics) / len(all_frame_metrics)
            for t in range(T)
        ]

    return avg_metrics, avg_frame_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/eval')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_vis', type=int, default=20)
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    config['dataset'] = dataset_config['dataset']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建测试集
    test_dataset = build_dataset(config['dataset'], split='test')
    test_loader, _ = get_dataloader(test_dataset, args.batch_size, is_train=False)

    # 构建模型并加载权重
    model = build_model(config)
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model params: {model.get_num_params() / 1e6:.2f}M")
    print(f"Test dataset: {len(test_dataset)} samples")

    # 评估
    metric_calc = MetricCalculator(device)
    avg_metrics, frame_metrics = evaluate(
        model, test_loader, metric_calc, device, output_dir, args.num_vis
    )

    # 打印结果
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in avg_metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 50)

    # 保存结果
    results = {
        'config': args.config,
        'dataset': args.dataset_config,
        'checkpoint': args.checkpoint,
        'metrics': avg_metrics,
        'frame_metrics': frame_metrics,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制逐帧性能衰减曲线
    plot_metrics_over_time(frame_metrics, str(output_dir / 'metrics_over_time.png'))

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
