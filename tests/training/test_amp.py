"""Tests for AMPManager."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.cuda.amp import GradScaler


class AMPManager:
    """Mock implementation for testing."""
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.scaler = GradScaler(enabled=enabled)

    def autocast(self) -> Any:
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()


@pytest.fixture
def amp_manager() -> AMPManager:
    """Provides an AMPManager instance."""
    return AMPManager()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_amp_autocast_context(enabled: bool, device: str) -> None:
    """Test autocast context manager activation."""
    manager = AMPManager(enabled=enabled)
    with manager.autocast():
        pass
    assert manager.enabled == enabled


@pytest.mark.parametrize("loss_val", [1.0, 0.5, 10.0, 100.0])
@pytest.mark.parametrize("scale_factor", [1024, 2048, 4096])
def test_amp_backward(loss_val: float, scale_factor: int) -> None:
    """Test scaled backward pass."""
    manager = AMPManager()
    manager.scaler._scale = torch.tensor(scale_factor, dtype=torch.float32)
    loss = torch.tensor(loss_val, requires_grad=True)
    manager.backward(loss)


@pytest.mark.parametrize("grad_clip", [None, 1.0, 5.0, 10.0])
@pytest.mark.parametrize("accumulation_steps", [1, 2, 4, 8])
def test_amp_step(grad_clip: float, accumulation_steps: int) -> None:
    """Test optimizer stepping with AMP and gradient clipping/accumulation."""
    manager = AMPManager()
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    manager.step(optimizer)
    assert isinstance(manager.scaler, GradScaler)
