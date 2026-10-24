"""torch.distributed helpers (no-ops in single-process runs). Launch multi-GPU runs with torchrun."""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def local_rank():
    return int(os.environ.get('LOCAL_RANK', -1))


def init_distributed():
    """Initialize the process group if launched by torchrun. Returns (rank, world_size)."""
    if local_rank() < 0:
        return 0, 1
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    return dist.get_rank(), dist.get_world_size()


def world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if world_size() > 1:
        dist.barrier()


def all_gather(t):
    """Concatenate a tensor from all processes along dim 0."""
    if world_size() <= 1:
        return t
    out = [torch.zeros_like(t) for _ in range(world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, 0)


def broadcast_object(obj):
    if world_size() <= 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, 0)
    return box[0]


def cleanup():
    if world_size() > 1:
        dist.destroy_process_group()
