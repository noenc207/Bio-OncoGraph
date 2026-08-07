"""Tests for Sliced Wasserstein Distance (SWD) Loss.

This module contains tests for SWDLoss.
"""
import numpy as np
import pytest
from bio_oncograph.ai.losses.swd import SWDLoss


def test_swd_loss_basic() -> None:
    """Test basic computation of SWDLoss."""
    loss_fn = SWDLoss(num_projections=50, random_state=42)
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 10)
    loss = loss_fn(x, y)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss >= 0.0


def test_swd_loss_identical_distributions() -> None:
    """Test SWD between two identical distributions is ~0."""
    loss_fn = SWDLoss(num_projections=100, random_state=42)
    x = np.random.randn(100, 10)
    # Using the exact same distribution
    loss = loss_fn(x, x)
    assert loss < 1e-5


def test_swd_loss_distinct_distributions() -> None:
    """Test SWD between distinct distributions is > 0."""
    loss_fn = SWDLoss(num_projections=50, random_state=42)
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 10) + 10.0  # Shift mean significantly
    loss = loss_fn(x, y)
    assert loss > 1.0


def test_swd_loss_deterministic() -> None:
    """Ensure deterministic output with seeded random projections."""
    loss_fn1 = SWDLoss(num_projections=50, random_state=123)
    loss_fn2 = SWDLoss(num_projections=50, random_state=123)
    x = np.random.randn(50, 5)
    y = np.random.randn(50, 5)
    
    loss1 = loss_fn1(x, y)
    loss2 = loss_fn2(x, y)
    
    assert loss1 == loss2


def test_swd_loss_different_seeds() -> None:
    """Test SWD with different seeds yields slightly different projections."""
    loss_fn1 = SWDLoss(num_projections=10, random_state=123)
    loss_fn2 = SWDLoss(num_projections=10, random_state=456)
    x = np.random.randn(50, 5)
    y = np.random.randn(50, 5)
    
    loss1 = loss_fn1(x, y)
    loss2 = loss_fn2(x, y)
    
    assert loss1 != loss2


def test_swd_loss_numerical_stability() -> None:
    """Test numerical stability with large values."""
    loss_fn = SWDLoss(num_projections=10, random_state=42)
    x = np.random.randn(10, 5) * 1e6
    y = np.random.randn(10, 5) * 1e6
    loss = loss_fn(x, y)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_swd_loss_different_batch_sizes() -> None:
    """Test SWD with different batch sizes for x and y."""
    loss_fn = SWDLoss(num_projections=50, random_state=42)
    x = np.random.randn(100, 10)
    y = np.random.randn(50, 10)
    loss = loss_fn(x, y)
    assert not np.isnan(loss)
