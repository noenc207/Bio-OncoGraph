"""Tests for CalTopoSurvLoss.

This module contains tests for CalTopoSurvLoss.
"""
import numpy as np
import pytest
from bio_oncograph.ai.losses.caltoposurv import CalTopoSurvLoss


def test_caltoposurv_loss_basic() -> None:
    """Test basic computation of CalTopoSurvLoss."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.5, -0.2, 1.1, 0.0])
    survival_times = np.array([15, 25, 5, 30])
    events = np.array([1, 0, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert isinstance(loss, float)
    assert not np.isnan(loss)


def test_caltoposurv_loss_all_events() -> None:
    """Test CalTopoSurvLoss when all patients experience the event."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.1, 0.2, 0.3])
    survival_times = np.array([10, 20, 30])
    events = np.array([1, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)


def test_caltoposurv_loss_all_censored() -> None:
    """Test CalTopoSurvLoss when all patients are censored."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.1, 0.2, 0.3])
    survival_times = np.array([10, 20, 30])
    events = np.array([0, 0, 0])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)


def test_caltoposurv_loss_edge_case_identical_times() -> None:
    """Test CalTopoSurvLoss with identical survival times."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.1, 0.2, 0.3, 0.4])
    survival_times = np.array([10, 10, 10, 10])
    events = np.array([1, 0, 1, 0])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)


def test_caltoposurv_loss_edge_case_identical_scores() -> None:
    """Test CalTopoSurvLoss with identical risk scores."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.5, 0.5, 0.5])
    survival_times = np.array([10, 20, 30])
    events = np.array([1, 1, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)


def test_caltoposurv_loss_numerical_stability() -> None:
    """Test numerical stability with extreme risk scores."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([100.0, -100.0, 50.0])
    survival_times = np.array([10, 20, 30])
    events = np.array([1, 0, 1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_caltoposurv_loss_single_patient() -> None:
    """Test CalTopoSurvLoss with a single patient."""
    loss_fn = CalTopoSurvLoss()
    risk_scores = np.array([0.5])
    survival_times = np.array([10])
    events = np.array([1])
    loss = loss_fn(risk_scores, survival_times, events)
    assert not np.isnan(loss)
