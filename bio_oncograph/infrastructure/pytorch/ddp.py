import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Any

class DDPManager:
    @staticmethod
    def setup(backend: str = "nccl") -> None:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

    @staticmethod
    def cleanup() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def wrap_model(model: torch.nn.Module, **kwargs) -> DistributedDataParallel:
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")
        device_id = int(os.environ.get("LOCAL_RANK", 0))
        model = model.to(device_id)
        return DistributedDataParallel(model, device_ids=[device_id], **kwargs)

    @staticmethod
    def is_master() -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    @staticmethod
    def get_world_size() -> int:
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @staticmethod
    def get_rank() -> int:
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def sync_metrics(tensor: torch.Tensor, op: Any = dist.ReduceOp.AVG) -> torch.Tensor:
        if not dist.is_initialized():
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor
