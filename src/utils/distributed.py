"""分布式训练工具"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_distributed(backend='nccl'):
    """初始化分布式训练环境"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model_ddp(model, local_rank, find_unused_parameters=False):
    """包装模型为 DDP"""
    model = model.cuda(local_rank)
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=find_unused_parameters)
    return model


def get_dataloader(dataset, batch_size, num_workers=4, is_train=True):
    """创建支持分布式的 DataLoader"""
    sampler = None
    shuffle = is_train

    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        shuffle = False

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader, sampler


def reduce_tensor(tensor):
    """跨进程平均 tensor"""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def is_main_process():
    """判断是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0
