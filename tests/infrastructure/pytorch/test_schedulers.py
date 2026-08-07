"""Tests for Scheduler factory and functionality."""

import pytest
import torch
import torch.nn as nn
from typing import Any

def create_scheduler(name: str, optimizer: torch.optim.Optimizer, **kwargs: Any) -> torch.optim.lr_scheduler.LRScheduler:
    """Dummy scheduler factory for testing."""
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

@pytest.fixture
def dummy_optimizer() -> torch.optim.Optimizer:
    """Fixture providing a dummy optimizer."""
    model = nn.Linear(10, 5)
    return torch.optim.SGD(model.parameters(), lr=0.1)

def test_factory_step(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test creating StepLR scheduler."""
    scheduler = create_scheduler("step", dummy_optimizer, step_size=10, gamma=0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1

def test_factory_cosine(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test creating CosineAnnealingLR scheduler."""
    scheduler = create_scheduler("cosine", dummy_optimizer, T_max=50)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 50

def test_factory_invalid(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test invalid scheduler name raises error."""
    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler("plateau", dummy_optimizer)

def test_scheduler_step(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test scheduler updates learning rate after step."""
    scheduler = create_scheduler("step", dummy_optimizer, step_size=1, gamma=0.5)
    initial_lr = dummy_optimizer.param_groups[0]["lr"]
    scheduler.step()
    new_lr = dummy_optimizer.param_groups[0]["lr"]
    assert new_lr == initial_lr * 0.5

def test_scheduler_state_dict(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test getting scheduler state dict."""
    scheduler = create_scheduler("step", dummy_optimizer, step_size=10, gamma=0.1)
    state = scheduler.state_dict()
    assert "step_size" in state
    assert "gamma" in state
    assert state["step_size"] == 10

def test_scheduler_load_state_dict(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test loading scheduler state dict restores state."""
    scheduler1 = create_scheduler("step", dummy_optimizer, step_size=10, gamma=0.1)
    scheduler1.step()
    
    model2 = nn.Linear(10, 5)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    scheduler2 = create_scheduler("step", opt2, step_size=10, gamma=0.1)
    
    scheduler2.load_state_dict(scheduler1.state_dict())
    assert scheduler1.last_epoch == scheduler2.last_epoch

def test_scheduler_last_epoch(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test scheduler tracks last epoch correctly."""
    scheduler = create_scheduler("step", dummy_optimizer, step_size=10, gamma=0.1)
    assert scheduler.last_epoch == 0
    scheduler.step()
    assert scheduler.last_epoch == 1

def test_scheduler_get_last_lr(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test get_last_lr returns correct value."""
    scheduler = create_scheduler("step", dummy_optimizer, step_size=1, gamma=0.5)
    scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.05

def test_scheduler_multiple_param_groups() -> None:
    """Test scheduler with multiple parameter groups."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
    params = [
        {"params": model[0].parameters(), "lr": 0.1},
        {"params": model[1].parameters(), "lr": 0.01}
    ]
    opt = torch.optim.SGD(params)
    scheduler = create_scheduler("step", opt, step_size=1, gamma=0.1)
    
    scheduler.step()
    assert opt.param_groups[0]["lr"] == 0.01
    assert pytest.approx(opt.param_groups[1]["lr"]) == 0.001

def test_scheduler_cosine_decay(dummy_optimizer: torch.optim.Optimizer) -> None:
    """Test cosine annealing decay behavior."""
    scheduler = create_scheduler("cosine", dummy_optimizer, T_max=10, eta_min=0.0)
    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # Check that LR generally decreases over the cycle
    assert lrs[-1] < lrs[0]
