"""Tests for survival loss functions.

This module contains tests for CoxLoss.
"""
import numpy as np
import pytest
from bio_oncograph.ai.losses.survival import CoxLoss


def test_cox_loss_basic() -> None:
    """Test basic Cox loss computation."""
    loss_fn = CoxLoss()
    risk_scores = np.array([0.5, 1.2, -0.3, 2.0])
    survival_times = np.array([10, 20, 30, 40])
    events = np.array([1, 1, 0, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_cox_loss_sorted_times() -> None:
    """Test Cox loss with naturally sorted survival times."""
    loss_fn = CoxLoss()
    risk_scores = np.array([0.1, 0.2, 0.3])
    survival_times = np.array([10, 20, 30])
    events = np.array([1, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)


def test_cox_loss_unsorted_times() -> None:
    """Test Cox loss with unsorted survival times."""
    loss_fn = CoxLoss()
    risk_scores_sorted = np.array([0.1, 0.2, 0.3])
    times_sorted = np.array([10, 20, 30])
    events_sorted = np.array([1, 1, 1])
    
    risk_scores_unsorted = np.array([0.2, 0.3, 0.1])
    times_unsorted = np.array([20, 30, 10])
    events_unsorted = np.array([1, 1, 1])
    
    loss_sorted = loss_fn(risk_scores_sorted, times_sorted, events_sorted)
    loss_unsorted = loss_fn(risk_scores_unsorted, times_unsorted, events_unsorted)
    
    np.testing.assert_allclose(loss_sorted, loss_unsorted, rtol=1e-5)


def test_cox_loss_all_censored() -> None:
    """Test Cox loss when all events are censored."""
    loss_fn = CoxLoss()
    risk_scores = np.array([0.5, 1.2, -0.3])
    survival_times = np.array([10, 20, 30])
    events = np.array([0, 0, 0])
    loss = loss_fn(risk_scores, survival_times, events)
    assert loss == 0.0 or not np.isnan(loss)


def test_cox_loss_numerical_stability() -> None:
    """Test numerical stability with large risk scores."""
    loss_fn = CoxLoss()
    risk_scores = np.array([50.0, 100.0, -100.0])
    survival_times = np.array([10, 20, 30])
    events = np.array([1, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_cox_loss_ties() -> None:
    """Test Cox loss with tied survival times."""
    loss_fn = CoxLoss()
    risk_scores = np.array([0.1, 0.2, 0.3])
    survival_times = np.array([10, 10, 30])
    events = np.array([1, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_cox_loss_single_event() -> None:
    """Test Cox loss with a single event."""
    loss_fn = CoxLoss()
    risk_scores = np.array([0.1, 0.2, 0.3])
    survival_times = np.array([10, 20, 30])
    events = np.array([0, 1, 0])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)
