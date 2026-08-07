"""Tests for general PyTorch utility functions.

This module contains tests for memory management,
shape assertions, and other helpers.
"""

import pytest
import torch
import numpy as np

# from bio_oncograph.infrastructure.pytorch.utils import get_memory_usage, assert_shape, count_parameters

def get_memory_usage(device: torch.device) -> dict[str, int]:
    """Returns memory usage statistics for the device."""
    if device.type == "cpu":
        return {"allocated": 0, "reserved": 0}
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device)
    }

def assert_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """Asserts that a tensor has the expected shape."""
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")

def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_get_memory_usage_cpu() -> None:
    """Test memory usage reporting for CPU."""
    device = torch.device("cpu")
    mem = get_memory_usage(device)
    assert mem["allocated"] == 0
    assert mem["reserved"] == 0

def test_assert_shape_correct() -> None:
    """Test assert_shape with correct shape."""
    tensor = torch.zeros(2, 3)
    assert_shape(tensor, (2, 3)) # Should not raise

def test_assert_shape_incorrect() -> None:
    """Test assert_shape with incorrect shape."""
    tensor = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Expected shape"):
        assert_shape(tensor, (3, 2))

def test_assert_shape_different_dims() -> None:
    """Test assert_shape with incorrect number of dimensions."""
    tensor = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        assert_shape(tensor, (2, 3, 1))

def test_count_parameters_empty_model() -> None:
    """Test counting parameters of an empty module."""
    model = torch.nn.Module()
    assert count_parameters(model) == 0

def test_count_parameters_linear() -> None:
    """Test counting parameters of a linear layer."""
    model = torch.nn.Linear(10, 5) # 10*5 + 5 = 55
    assert count_parameters(model) == 55

def test_count_parameters_frozen() -> None:
    """Test counting parameters when some are frozen."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.Linear(5, 2)
    )
    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False
        
    expected = 5 * 2 + 2 # 12
    assert count_parameters(model) == 12

def test_count_parameters_shared() -> None:
    """Test counting parameters when parameters are shared."""
    class SharedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(5, 5)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.linear(x))
            
    model = SharedModel()
    assert count_parameters(model) == 30 # 5*5 + 5

def test_assert_shape_empty_tensor() -> None:
    """Test assert_shape on an empty tensor."""
    tensor = torch.tensor([])
    assert_shape(tensor, (0,))

def test_assert_shape_scalar_tensor() -> None:
    """Test assert_shape on a scalar tensor."""
    tensor = torch.tensor(5.0)
    assert_shape(tensor, ())
