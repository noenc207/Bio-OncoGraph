"""Tests for classification loss functions.

This module contains tests for CrossEntropyLoss and BinaryCrossEntropyLoss.
"""
import numpy as np
import pytest
from bio_oncograph.ai.losses.classification import CrossEntropyLoss, BinaryCrossEntropyLoss


def test_bce_loss_basic() -> None:
    """Test basic computation of BinaryCrossEntropyLoss."""
    loss_fn = BinaryCrossEntropyLoss()
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, float)
    assert not np.isnan(loss)
    assert loss > 0


def test_bce_loss_exact_value() -> None:
    """Test BCE loss against a known hand-calculated value."""
    loss_fn = BinaryCrossEntropyLoss()
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.5, 0.5])
    loss = loss_fn(y_pred, y_true)
    expected_loss = -np.log(0.5)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_bce_epsilon_protection_ones() -> None:
    """Test BCE loss when predictions are exactly 1.0 (epsilon protection)."""
    loss_fn = BinaryCrossEntropyLoss(epsilon=1e-7)
    y_true = np.array([1, 0])
    y_pred = np.array([1.0, 1.0])
    loss = loss_fn(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_bce_epsilon_protection_zeros() -> None:
    """Test BCE loss when predictions are exactly 0.0 (epsilon protection)."""
    loss_fn = BinaryCrossEntropyLoss(epsilon=1e-7)
    y_true = np.array([1, 0])
    y_pred = np.array([0.0, 0.0])
    loss = loss_fn(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_bce_loss_perfect_prediction() -> None:
    """Test BCE loss with perfect predictions."""
    loss_fn = BinaryCrossEntropyLoss()
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1.0, 0.0, 1.0])
    loss = loss_fn(y_pred, y_true)
    assert loss < 1e-4


def test_ce_loss_basic() -> None:
    """Test basic computation of CrossEntropyLoss."""
    loss_fn = CrossEntropyLoss()
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_ce_loss_exact_value() -> None:
    """Test CE loss against a known calculated value."""
    loss_fn = CrossEntropyLoss()
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
    loss = loss_fn(y_pred, y_true)
    expected_loss = -np.log(0.5)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)


def test_ce_epsilon_protection() -> None:
    """Test CE loss with extreme predictions."""
    loss_fn = CrossEntropyLoss(epsilon=1e-7)
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    loss = loss_fn(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss < 1e-4


def test_ce_epsilon_protection_zeros() -> None:
    """Test CE loss with zero probabilities for true classes."""
    loss_fn = CrossEntropyLoss(epsilon=1e-7)
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.0, 1.0], [1.0, 0.0]])
    loss = loss_fn(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert loss > 10.0  # Should be a large positive number due to log(epsilon)
