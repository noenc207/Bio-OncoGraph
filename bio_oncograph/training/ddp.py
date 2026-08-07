"""Distributed Data Parallel (DDP) management module."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DDPManager:
    """Manages Distributed Data Parallel (DDP) environment setup, model wrapping, and cleanup."""

    def __init__(self, backend: str = "nccl") -> None:
        """Initializes the DDPManager.

        Args:
            backend: The distributed backend to use (e.g., 'nccl', 'gloo').
        """
        self.backend = backend
        self.is_initialized = False
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

    def setup(self) -> None:
        """Sets up the distributed process group."""
        if self.world_size > 1 and not dist.is_initialized():
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
            self.is_initialized = True

    def cleanup(self) -> None:
        """Destroys the distributed process group."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False

    def wrap_model(self, model: torch.nn.Module, sync_batchnorm: bool = False) -> torch.nn.Module:
        """Wraps the model with DistributedDataParallel.

        Args:
            model: The PyTorch model to wrap.
            sync_batchnorm: Whether to convert BatchNorm layers to SyncBatchNorm.

        Returns:
            The DDP-wrapped model, or the original model if world_size is 1.
        """
        if self.world_size <= 1:
            return model.to(self.local_rank)

        model = model.to(self.local_rank)
        if sync_batchnorm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        return DistributedDataParallel(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def sync_tensor(self, tensor: torch.Tensor, reduce_op: str = "mean") -> torch.Tensor:
        """Synchronizes a tensor across all distributed processes.

        Args:
            tensor: The tensor to synchronize.
            reduce_op: Reduction operation ('sum', 'mean').

        Returns:
            The synchronized tensor.
        """
        if self.world_size <= 1:
            return tensor

        synced_tensor = tensor.clone().detach()
        dist.all_reduce(synced_tensor, op=dist.ReduceOp.SUM)
        
        if reduce_op == "mean":
            synced_tensor /= self.world_size
            
        return synced_tensor
