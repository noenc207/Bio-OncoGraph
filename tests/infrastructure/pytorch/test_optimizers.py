"""Tests for Optimizer factory and functionality."""

import pytest
import torch
import torch.nn as nn
from typing import Any

def create_optimizer(name: str, params: Any, **kwargs: Any) -> torch.optim.Optimizer:
    """Dummy optimizer factory for testing."""
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def test_factory_adam() -> None:
    """Test creating Adam optimizer sets correct defaults."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("adam", model.parameters(), lr=0.01)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.param_groups[0]["lr"] == 0.01

def test_factory_sgd() -> None:
    """Test creating SGD optimizer sets correct defaults."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("sgd", model.parameters(), lr=0.1, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["lr"] == 0.1
    assert opt.param_groups[0]["momentum"] == 0.9

def test_factory_invalid() -> None:
    """Test invalid optimizer name raises ValueError."""
    model = nn.Linear(10, 5)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        create_optimizer("rmsprop", model.parameters())

def test_optimizer_step() -> None:
    """Test optimizer step updates model parameters."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("sgd", model.parameters(), lr=0.1)
    weight_before = model.weight.clone()
    
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    
    assert not torch.equal(model.weight, weight_before)

def test_optimizer_zero_grad() -> None:
    """Test zeroing gradients clears parameter grads."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("sgd", model.parameters(), lr=0.1)
    
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    
    assert model.weight.grad is not None
    opt.zero_grad()
    assert torch.all(model.weight.grad == 0)

def test_optimizer_state_dict() -> None:
    """Test getting optimizer state dict returns correct keys."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("adam", model.parameters(), lr=0.01)
    state = opt.state_dict()
    assert "state" in state
    assert "param_groups" in state

def test_optimizer_load_state_dict() -> None:
    """Test loading optimizer state dict restores state."""
    model1 = nn.Linear(10, 5)
    opt1 = create_optimizer("adam", model1.parameters(), lr=0.01)
    
    x = torch.randn(2, 10)
    loss = model1(x).sum()
    loss.backward()
    opt1.step()
    
    model2 = nn.Linear(10, 5)
    opt2 = create_optimizer("adam", model2.parameters(), lr=0.01)
    opt2.load_state_dict(opt1.state_dict())
    
    assert opt1.state_dict() == opt2.state_dict()

def test_optimizer_param_groups() -> None:
    """Test multiple parameter groups with different learning rates."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
    params = [
        {"params": model[0].parameters(), "lr": 0.01},
        {"params": model[1].parameters(), "lr": 0.001}
    ]
    opt = create_optimizer("sgd", params)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 0.01
    assert opt.param_groups[1]["lr"] == 0.001

def test_optimizer_numerical_stability() -> None:
    """Test optimizer behavior with small gradients avoids NaNs."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("adam", model.parameters(), lr=1e-5)
    x = torch.randn(2, 10) * 1e-5
    loss = model(x).sum()
    loss.backward()
    opt.step()
    assert not torch.isnan(model.weight).any()

def test_optimizer_weight_decay() -> None:
    """Test weight decay functionality configures correctly."""
    model = nn.Linear(10, 5)
    opt = create_optimizer("sgd", model.parameters(), lr=0.1, weight_decay=1e-4)
    assert opt.param_groups[0]["weight_decay"] == 1e-4
