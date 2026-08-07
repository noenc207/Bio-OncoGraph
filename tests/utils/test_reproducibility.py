"""
Unit tests for bio_oncograph.utils.reproducibility.
"""

import os
import random

import numpy as np
import pytest
import torch

from bio_oncograph.utils.reproducibility import seed_everything


def test_seed_everything_returns_seed() -> None:
    """seed_everything echoes the seed value."""
    result = seed_everything(123)
    assert result == 123


def test_seed_everything_python_random() -> None:
    """Python random module is deterministic after seeding."""
    seed_everything(42)
    a = random.random()
    seed_everything(42)
    b = random.random()
    assert a == b


def test_seed_everything_numpy() -> None:
    """NumPy RNG is deterministic after seeding."""
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_seed_everything_torch() -> None:
    """PyTorch CPU RNG is deterministic after seeding."""
    seed_everything(42)
    a = torch.randn(5)
    seed_everything(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_seed_everything_cudnn_flags() -> None:
    """cuDNN deterministic flags are set correctly."""
    seed_everything(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_everything_workers_env_var() -> None:
    """PL_GLOBAL_SEED environment variable is set when workers=True."""
    seed_everything(99, workers=True)
    assert os.environ.get("PL_GLOBAL_SEED") == "99"


def test_seed_everything_no_workers_env_var() -> None:
    """PL_GLOBAL_SEED is not set when workers=False."""
    # Clear it first
    os.environ.pop("PL_GLOBAL_SEED", None)
    seed_everything(99, workers=False)
    assert os.environ.get("PL_GLOBAL_SEED") is None


def test_seed_everything_negative_raises() -> None:
    """Negative seed raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
