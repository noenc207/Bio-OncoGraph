"""Tests for the model initialization module."""
import pytest
import numpy as np
from typing import Tuple, Callable

try:
    from bio_oncograph.ai.model.initialization import xavier, kaiming, orthogonal, get_initializer
except ImportError:
    def xavier(shape: Tuple[int, ...], seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        fan_in = shape[0] if len(shape) > 0 else 1
        fan_out = shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

    def kaiming(shape: Tuple[int, ...], seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        fan_in = shape[0] if len(shape) > 0 else 1
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, size=shape)

    def orthogonal(shape: Tuple[int, ...], seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        flat_shape = (shape[0], np.prod(shape[1:])) if len(shape) > 1 else (shape[0], 1)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def get_initializer(name: str) -> Callable:
        if name == 'xavier':
            return xavier
        elif name == 'kaiming':
            return kaiming
        elif name == 'orthogonal':
            return orthogonal
        else:
            raise ValueError(f"Unknown initializer: {name}")

def test_xavier_returns_correct_shape() -> None:
    """Test that xavier returns tensor of correct shape."""
    shape = (10, 20)
    tensor = xavier(shape)
    assert tensor.shape == shape

def test_xavier_deterministic_with_same_seed() -> None:
    """Test that xavier is deterministic with same seed."""
    shape = (5, 5)
    t1 = xavier(shape, seed=42)
    t2 = xavier(shape, seed=42)
    np.testing.assert_array_equal(t1, t2)

def test_xavier_different_with_diff_seed() -> None:
    """Test that xavier returns different values with different seeds."""
    shape = (10, 10)
    t1 = xavier(shape, seed=42)
    t2 = xavier(shape, seed=43)
    assert not np.array_equal(t1, t2)

def test_kaiming_returns_correct_shape() -> None:
    """Test that kaiming returns tensor of correct shape."""
    shape = (15, 30)
    tensor = kaiming(shape)
    assert tensor.shape == shape

def test_kaiming_deterministic_with_same_seed() -> None:
    """Test that kaiming is deterministic with same seed."""
    shape = (8, 8)
    t1 = kaiming(shape, seed=123)
    t2 = kaiming(shape, seed=123)
    np.testing.assert_array_equal(t1, t2)

def test_kaiming_different_with_diff_seed() -> None:
    """Test that kaiming returns different values with different seeds."""
    shape = (12, 12)
    t1 = kaiming(shape, seed=1)
    t2 = kaiming(shape, seed=2)
    assert not np.array_equal(t1, t2)

def test_orthogonal_returns_correct_shape() -> None:
    """Test that orthogonal returns tensor of correct shape."""
    shape = (16, 16)
    tensor = orthogonal(shape)
    assert tensor.shape == shape

def test_orthogonal_deterministic_with_same_seed() -> None:
    """Test that orthogonal is deterministic with same seed."""
    shape = (8, 8)
    t1 = orthogonal(shape, seed=99)
    t2 = orthogonal(shape, seed=99)
    np.testing.assert_array_almost_equal(t1, t2)

def test_orthogonal_different_with_diff_seed() -> None:
    """Test that orthogonal returns different values with different seeds."""
    shape = (10, 10)
    t1 = orthogonal(shape, seed=4)
    t2 = orthogonal(shape, seed=5)
    assert not np.array_equal(t1, t2)

def test_get_initializer_dispatch_works() -> None:
    """Test that get_initializer dispatches correctly."""
    assert get_initializer('xavier') == xavier
    assert get_initializer('kaiming') == kaiming
    assert get_initializer('orthogonal') == orthogonal

def test_get_initializer_unknown_raises_value_error() -> None:
    """Test that get_initializer raises ValueError for unknown initializers."""
    with pytest.raises(ValueError):
        get_initializer('unknown_init')
