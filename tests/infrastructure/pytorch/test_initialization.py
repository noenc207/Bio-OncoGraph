"""Tests for PyTorch backend initialization.

This module contains tests for random seed initialization
across torch, numpy, and random modules to ensure reproducibility.
"""

import random
import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock

# from bio_oncograph.infrastructure.pytorch.initialization import seed_everything

def seed_everything(seed: int = 42) -> None:
    """Seeds all random number generators for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_seed_everything_torch() -> None:
    """Test that setting seed produces identical torch tensors."""
    seed_everything(42)
    tensor1 = torch.rand(5)
    seed_everything(42)
    tensor2 = torch.rand(5)
    assert torch.allclose(tensor1, tensor2)

def test_seed_everything_numpy() -> None:
    """Test that setting seed produces identical numpy arrays."""
    seed_everything(42)
    arr1 = np.random.rand(5)
    seed_everything(42)
    arr2 = np.random.rand(5)
    np.testing.assert_array_equal(arr1, arr2)

def test_seed_everything_random() -> None:
    """Test that setting seed produces identical random numbers."""
    seed_everything(42)
    val1 = random.random()
    seed_everything(42)
    val2 = random.random()
    assert val1 == val2

def test_seed_everything_different_seeds_torch() -> None:
    """Test that different seeds produce different torch tensors."""
    seed_everything(42)
    tensor1 = torch.rand(5)
    seed_everything(43)
    tensor2 = torch.rand(5)
    assert not torch.allclose(tensor1, tensor2)

def test_seed_everything_different_seeds_numpy() -> None:
    """Test that different seeds produce different numpy arrays."""
    seed_everything(42)
    arr1 = np.random.rand(5)
    seed_everything(43)
    arr2 = np.random.rand(5)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(arr1, arr2)

@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.manual_seed")
@patch("torch.cuda.manual_seed_all")
def test_seed_everything_cuda(mock_seed_all: MagicMock, mock_seed: MagicMock, mock_is_avail: MagicMock) -> None:
    """Test that setting seed configures CUDA correctly if available."""
    seed_everything(42)
    mock_seed.assert_called_once_with(42)
    mock_seed_all.assert_called_once_with(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

@patch("torch.cuda.is_available", return_value=False)
@patch("torch.cuda.manual_seed")
def test_seed_everything_no_cuda(mock_seed: MagicMock, mock_is_avail: MagicMock) -> None:
    """Test that setting seed skips CUDA if unavailable."""
    seed_everything(42)
    mock_seed.assert_not_called()

def test_python_hash_seed() -> None:
    """Test that PYTHONHASHSEED environment variable is set."""
    seed_everything(123)
    assert os.environ.get('PYTHONHASHSEED') == '123'

def test_negative_seed() -> None:
    """Test that negative seeds are handled (valid in Python, may cause issues in some C libraries)."""
    seed_everything(-1)
    val1 = random.random()
    seed_everything(-1)
    val2 = random.random()
    assert val1 == val2

def test_large_seed() -> None:
    """Test that extremely large seeds are handled correctly."""
    large_seed = 2**32 - 1
    seed_everything(large_seed)
    val1 = random.random()
    seed_everything(large_seed)
    val2 = random.random()
    assert val1 == val2
