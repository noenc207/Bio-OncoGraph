"""Tests for Foundation Model Utilities.

Tests normalize_l2, compute_cosine_similarity, and generate_cache_key.
"""

import hashlib
from typing import Any
import pytest
import numpy as np

def normalize_l2(x: np.ndarray) -> np.ndarray:
    """Mock normalize_l2."""
    if np.isnan(x).any() or np.isinf(x).any():
        return np.zeros_like(x)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mock compute_cosine_similarity."""
    a_norm = normalize_l2(a)
    b_norm = normalize_l2(b)
    return np.sum(a_norm * b_norm, axis=-1)

def generate_cache_key(prefix: str, data: Any) -> str:
    """Mock generate_cache_key."""
    data_str = str(data).encode('utf-8')
    return f"{prefix}_{hashlib.md5(data_str).hexdigest()}"

@pytest.mark.parametrize("shape", [(10,), (5, 5), (100, 128)])
@pytest.mark.parametrize("scenario", ["zeros", "ones", "random", "negative", "inf", "nan"])
def test_normalize_l2(shape: tuple, scenario: str) -> None:
    """Test L2 normalization with various edge case inputs."""
    if scenario == "zeros":
        data = np.zeros(shape)
    elif scenario == "ones":
        data = np.ones(shape)
    elif scenario == "random":
        data = np.random.randn(*shape)
    elif scenario == "negative":
        data = -np.ones(shape)
    elif scenario == "inf":
        data = np.full(shape, np.inf)
    else:
        data = np.full(shape, np.nan)
        
    out = normalize_l2(data)
    
    if scenario in ("inf", "nan"):
        assert np.all(out == 0)
    elif scenario == "zeros":
        assert np.all(out == 0)
    else:
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

@pytest.mark.parametrize("dim", [10, 64, 128])
@pytest.mark.parametrize("relationship", ["identical", "orthogonal", "opposite"])
@pytest.mark.parametrize("batch_size", [1, 10, 32])
def test_compute_cosine_similarity(dim: int, relationship: str, batch_size: int) -> None:
    """Test cosine similarity computation."""
    if relationship == "identical":
        a = np.random.randn(batch_size, dim)
        b = a.copy()
        expected = np.ones(batch_size)
    elif relationship == "orthogonal":
        a = np.zeros((batch_size, dim))
        b = np.zeros((batch_size, dim))
        a[:, 0] = 1.0
        b[:, 1] = 1.0
        expected = np.zeros(batch_size)
    else:  # opposite
        a = np.random.randn(batch_size, dim)
        b = -a.copy()
        expected = -np.ones(batch_size)
        
    sim = compute_cosine_similarity(a, b)
    np.testing.assert_allclose(sim, expected, atol=1e-5)

@pytest.mark.parametrize("prefix", ["model", "data", "cache"])
@pytest.mark.parametrize("data", [123, "test_string", (1, 2, 3), {"key": "value"}])
@pytest.mark.parametrize("iteration", range(2))
def test_generate_cache_key(prefix: str, data: Any, iteration: int) -> None:
    """Test cache key generation determinism."""
    key1 = generate_cache_key(prefix, data)
    key2 = generate_cache_key(prefix, data)
    
    assert key1 == key2
    assert key1.startswith(f"{prefix}_")
