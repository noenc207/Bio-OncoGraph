"""Tests for utility functions."""

import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Sets seed for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("seed", [0, 42, 1234, 9999, 100000])
@pytest.mark.parametrize("offset", [0, 1, 10])
def test_set_seed_determinism(seed: int, offset: int) -> None:
    """Test seed ensures deterministic random generation."""
    set_seed(seed + offset)
    val1_np = np.random.rand()
    val1_torch = torch.rand(1).item()
    val1_py = random.random()
    
    set_seed(seed + offset)
    val2_np = np.random.rand()
    val2_torch = torch.rand(1).item()
    val2_py = random.random()
    
    assert val1_np == val2_np
    assert val1_torch == val2_torch
    assert val1_py == val2_py


@pytest.mark.parametrize("in_features", [10, 50, 100])
@pytest.mark.parametrize("out_features", [2, 10, 20])
@pytest.mark.parametrize("bias", [True, False])
def test_count_parameters(in_features: int, out_features: int, bias: bool) -> None:
    """Test parameter counting."""
    model = nn.Linear(in_features, out_features, bias=bias)
    expected = in_features * out_features + (out_features if bias else 0)
    assert count_parameters(model) == expected


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("hidden", [16, 32, 64])
def test_count_parameters_frozen(freeze: bool, hidden: int) -> None:
    """Test parameter counting with frozen layers."""
    model = nn.Sequential(
        nn.Linear(10, hidden),
        nn.Linear(hidden, 2)
    )
    if freeze:
        for param in model[0].parameters():
            param.requires_grad = False
            
    total_params = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    
    if freeze:
        assert trainable < total_params
    else:
        assert trainable == total_params
