"""Tests for custom nn.Module wrappers."""

import pytest
import torch
import torch.nn as nn

class DummyModule(nn.Module):
    """Dummy module for testing."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize the dummy module."""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dummy module."""
        return self.linear(x)

class DummyWrapper(nn.Module):
    """Dummy wrapper module for testing."""

    def __init__(self, module: nn.Module) -> None:
        """Initialize the wrapper."""
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the wrapper."""
        return self.module(x) * 2.0

def test_dummy_module_init() -> None:
    """Test module initialization sets correct attributes."""
    module = DummyModule(10, 5)
    assert isinstance(module, nn.Module)
    assert module.linear.in_features == 10
    assert module.linear.out_features == 5

def test_dummy_module_forward() -> None:
    """Test module forward pass produces correct shape."""
    module = DummyModule(10, 5)
    x = torch.randn(2, 10)
    out = module(x)
    assert out.shape == (2, 5)

def test_wrapper_init() -> None:
    """Test wrapper initialization stores module."""
    module = DummyModule(10, 5)
    wrapper = DummyWrapper(module)
    assert wrapper.module is module

def test_wrapper_forward() -> None:
    """Test wrapper forward pass modifies output."""
    module = DummyModule(10, 5)
    wrapper = DummyWrapper(module)
    x = torch.randn(2, 10)
    out_module = module(x)
    out_wrapper = wrapper(x)
    assert torch.allclose(out_wrapper, out_module * 2.0)

def test_module_gradients() -> None:
    """Test that gradients are computed and propagated correctly."""
    module = DummyModule(10, 5)
    x = torch.randn(2, 10)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert module.linear.weight.grad is not None
    assert module.linear.bias.grad is not None

def test_module_device_transfer() -> None:
    """Test module moving to CPU device."""
    module = DummyModule(10, 5)
    module.to(torch.device("cpu"))
    assert next(module.parameters()).device.type == "cpu"

def test_module_state_dict() -> None:
    """Test state dict generation contains parameters."""
    module = DummyModule(10, 5)
    state = module.state_dict()
    assert "linear.weight" in state
    assert "linear.bias" in state

def test_module_load_state_dict() -> None:
    """Test loading state dict restores parameters."""
    module1 = DummyModule(10, 5)
    module2 = DummyModule(10, 5)
    module2.load_state_dict(module1.state_dict())
    assert torch.equal(module1.linear.weight, module2.linear.weight)

def test_module_eval_mode() -> None:
    """Test evaluation mode flag."""
    module = DummyModule(10, 5)
    module.eval()
    assert not module.training

def test_module_train_mode() -> None:
    """Test training mode flag."""
    module = DummyModule(10, 5)
    module.eval()
    module.train()
    assert module.training
