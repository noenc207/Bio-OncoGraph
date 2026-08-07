"""Tests for DDPManager."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist


class DDPManager:
    """Mock implementation for testing."""
    def __init__(self, backend: str = "nccl") -> None:
        self.backend = backend
        self.is_initialized = False

    def setup(self, rank: int, world_size: int) -> None:
        self.is_initialized = True

    def cleanup(self) -> None:
        self.is_initialized = False

    def sync_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


@pytest.fixture
def ddp_manager() -> DDPManager:
    """Provides a DDPManager instance."""
    return DDPManager()


@patch("torch.distributed.init_process_group")
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("backend", ["nccl", "gloo"])
def test_ddp_setup(mock_init: MagicMock, ddp_manager: DDPManager, rank: int, world_size: int, backend: str) -> None:
    """Test DDP setup initializes process group correctly."""
    ddp_manager.backend = backend
    ddp_manager.setup(rank, world_size)
    assert ddp_manager.is_initialized


@patch("torch.distributed.destroy_process_group")
@pytest.mark.parametrize("was_initialized", [True, False])
def test_ddp_cleanup(mock_destroy: MagicMock, ddp_manager: DDPManager, was_initialized: bool) -> None:
    """Test DDP cleanup destroys process group."""
    ddp_manager.is_initialized = was_initialized
    ddp_manager.cleanup()
    assert not ddp_manager.is_initialized


@patch("torch.distributed.all_reduce")
@pytest.mark.parametrize("tensor_shape", [(10,), (5, 5), (100, 100), (3, 224, 224)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sync_tensors(mock_all_reduce: MagicMock, ddp_manager: DDPManager, tensor_shape: tuple, dtype: torch.dtype) -> None:
    """Test tensor synchronization across processes."""
    tensor = torch.randn(tensor_shape, dtype=dtype)
    synced_tensor = ddp_manager.sync_tensors(tensor)
    assert synced_tensor.shape == tensor_shape
    assert synced_tensor.dtype == dtype
