"""Tests for scheduler creation and step logic."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


class SchedulerFactory:
    """Mock implementation for testing."""
    @staticmethod
    def create(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Any:
        if config["type"] == "step":
            return StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
        if config["type"] == "cosine":
            return CosineAnnealingLR(optimizer, T_max=config["t_max"])
        if config["type"] == "plateau":
            return ReduceLROnPlateau(optimizer, patience=config["patience"])
        raise ValueError("Unknown scheduler type")


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    """Provides a dummy optimizer."""
    model = torch.nn.Linear(10, 2)
    return Adam(model.parameters(), lr=0.1)


@pytest.mark.parametrize("step_size", [1, 5, 10, 50])
@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9, 1.0])
def test_create_step_lr(optimizer: torch.optim.Optimizer, step_size: int, gamma: float) -> None:
    """Test StepLR creation with varying parameters."""
    config = {"type": "step", "step_size": step_size, "gamma": gamma}
    scheduler = SchedulerFactory.create(optimizer, config)
    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == step_size
    assert scheduler.gamma == gamma


@pytest.mark.parametrize("t_max", [10, 50, 100, 200, 500])
@pytest.mark.parametrize("eta_min", [0.0, 1e-4, 1e-5])
def test_create_cosine_lr(optimizer: torch.optim.Optimizer, t_max: int, eta_min: float) -> None:
    """Test CosineAnnealingLR creation."""
    config = {"type": "cosine", "t_max": t_max}
    scheduler = SchedulerFactory.create(optimizer, config)
    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == t_max


@pytest.mark.parametrize("patience", [2, 5, 10])
@pytest.mark.parametrize("factor", [0.1, 0.5])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_create_plateau_lr(optimizer: torch.optim.Optimizer, patience: int, factor: float, mode: str) -> None:
    """Test ReduceLROnPlateau creation."""
    config = {"type": "plateau", "patience": patience}
    scheduler = SchedulerFactory.create(optimizer, config)
    assert isinstance(scheduler, ReduceLROnPlateau)
    assert scheduler.patience == patience


@pytest.mark.parametrize("invalid_type", ["unknown", "linear", "exponential", "none"])
def test_create_invalid_scheduler(optimizer: torch.optim.Optimizer, invalid_type: str) -> None:
    """Test factory raises error on invalid type."""
    config = {"type": invalid_type}
    with pytest.raises(ValueError):
        SchedulerFactory.create(optimizer, config)


@pytest.mark.parametrize("steps", [1, 5, 10])
def test_scheduler_step_execution(optimizer: torch.optim.Optimizer, steps: int) -> None:
    """Test scheduler step logic."""
    config = {"type": "step", "step_size": 1, "gamma": 0.1}
    scheduler = SchedulerFactory.create(optimizer, config)
    for _ in range(steps):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] < 0.1
