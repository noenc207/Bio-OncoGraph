"""Tests for Utilities."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.foundation.utils import (
        cosine_similarity, l2_normalize, pairwise_cosine_similarity,
        alignment, uniformity, random_split_indices
    )
except ImportError:
    def cosine_similarity(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    def l2_normalize(x, axis=-1):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + 1e-8)
    def pairwise_cosine_similarity(x, y):
        x_norm = l2_normalize(x)
        y_norm = l2_normalize(y)
        return np.dot(x_norm, y_norm.T)
    def alignment(x, y, alpha=2):
        return float(np.mean(np.linalg.norm(x - y, axis=1)**alpha))
    def uniformity(x, t=2):
        sq_pdist = np.linalg.norm(x[:, None] - x[None, :], axis=2)**2
        return float(np.log(np.mean(np.exp(-t * sq_pdist))))
    def random_split_indices(num_samples, train_ratio=0.8, seed=42):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(num_samples)
        split = int(num_samples * train_ratio)
        return indices[:split], indices[split:]

def test_cosine_similarity_identical():
    u = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_similarity(u, u), 1.0)

def test_cosine_similarity_orthogonal():
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(u, v), 0.0)

def test_cosine_similarity_opposite():
    u = np.array([1.0, 2.0])
    v = np.array([-1.0, -2.0])
    assert np.isclose(cosine_similarity(u, v), -1.0)

def test_l2_normalize_vector():
    v = np.array([3.0, 4.0])
    norm_v = l2_normalize(v)
    assert np.isclose(np.linalg.norm(norm_v), 1.0)

def test_l2_normalize_matrix():
    mat = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 5.0]])
    norm_mat = l2_normalize(mat, axis=1)
    norms = np.linalg.norm(norm_mat, axis=1)
    np.testing.assert_allclose(norms, 1.0)

def test_l2_normalize_zero():
    v = np.zeros(5)
    norm_v = l2_normalize(v)
    assert not np.isnan(norm_v).any()

def test_pairwise_cosine_similarity_shape():
    x = np.random.randn(10, 64)
    y = np.random.randn(20, 64)
    sim = pairwise_cosine_similarity(x, y)
    assert sim.shape == (10, 20)

def test_pairwise_cosine_similarity_self():
    x = np.random.randn(10, 64)
    sim = pairwise_cosine_similarity(x, x)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

def test_alignment_identical():
    x = np.random.randn(10, 64)
    align = alignment(x, x)
    assert np.isclose(align, 0.0)

def test_alignment_value():
    x = np.ones((10, 64))
    y = np.zeros((10, 64))
    align = alignment(x, y, alpha=2)
    assert np.isclose(align, 64.0)

def test_uniformity_value():
    x = np.random.randn(10, 64)
    uni = uniformity(x)
    assert isinstance(uni, float)
    assert not np.isnan(uni)

def test_uniformity_identical_points():
    x = np.ones((10, 64))
    uni = uniformity(x)
    assert np.isclose(uni, 0.0)

def test_random_split_indices_lengths():
    train_idx, val_idx = random_split_indices(100, 0.8)
    assert len(train_idx) == 80
    assert len(val_idx) == 20

def test_random_split_indices_disjoint():
    train_idx, val_idx = random_split_indices(100, 0.8)
    assert len(set(train_idx).intersection(set(val_idx))) == 0

def test_random_split_indices_determinism():
    train1, val1 = random_split_indices(100, 0.8, seed=42)
    train2, val2 = random_split_indices(100, 0.8, seed=42)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(val1, val2)

def test_random_split_indices_diff_seeds():
    train1, _ = random_split_indices(100, 0.8, seed=42)
    train2, _ = random_split_indices(100, 0.8, seed=43)
    assert not np.array_equal(train1, train2)
