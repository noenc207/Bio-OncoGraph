"""Tests for regularization loss functions.

This module contains tests for L1Loss, L2Loss, and WeightDecay.
"""
import numpy as np
import pytest
from bio_oncograph.ai.losses.regularization import L1Loss, L2Loss, WeightDecay


def test_l1_loss_basic() -> None:
    """Test basic L1 loss computation."""
    loss_fn = L1Loss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 4.0])
    loss = loss_fn(y_pred, y_true)
    expected_loss = np.mean(np.abs(y_pred - y_true))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_l1_loss_zero() -> None:
    """Test L1 loss when predictions match ground truth perfectly."""
    loss_fn = L1Loss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    loss = loss_fn(y_pred, y_true)
    assert loss == 0.0


def test_l2_loss_basic() -> None:
    """Test basic L2 loss computation."""
    loss_fn = L2Loss()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.0, 4.0])
    loss = loss_fn(y_pred, y_true)
    expected_loss = np.mean(np.square(y_pred - y_true))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_l2_loss_zero() -> None:
    """Test L2 loss when predictions match perfectly."""
    loss_fn = L2Loss()
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    loss = loss_fn(y_pred, y_true)
    assert loss == 0.0


def test_weight_decay_basic() -> None:
    """Test weight decay computation on a mock weights matrix."""
    loss_fn = WeightDecay(factor=0.1)
    weights = np.array([[1.0, -2.0], [3.0, 0.5]])
    loss = loss_fn(weights)
    # L2 penalty: sum(weights^2) / 2
    expected_loss = 0.1 * 0.5 * np.sum(np.square(weights))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_weight_decay_zero_weights() -> None:
    """Test weight decay with zero weights."""
    loss_fn = WeightDecay(factor=0.5)
    weights = np.zeros((3, 3))
    loss = loss_fn(weights)
    assert loss == 0.0


def test_weight_decay_different_factor() -> None:
    """Test weight decay with a different factor."""
    loss_fn = WeightDecay(factor=0.01)
    weights = np.array([2.0, 2.0])
    loss = loss_fn(weights)
    expected_loss = 0.01 * 0.5 * 8.0
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_regularization_stability() -> None:
    """Test numerical stability with large numbers."""
    loss_fn_l1 = L1Loss()
    loss_fn_l2 = L2Loss()
    y_true = np.array([1e6, -1e6])
    y_pred = np.array([2e6, -2e6])
    loss_l1 = loss_fn_l1(y_pred, y_true)
    loss_l2 = loss_fn_l2(y_pred, y_true)
    assert not np.isnan(loss_l1)
    assert not np.isnan(loss_l2)
