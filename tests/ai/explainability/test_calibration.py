"""Tests for calibration metrics."""

import numpy as np
import pytest

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        if i == 0:
            in_bin = in_bin | (y_prob == 0)
        prob_in_bin = in_bin.mean()
        if prob_in_bin > 0:
            acc = y_true[in_bin].mean()
            conf = y_prob[in_bin].mean()
            ece += prob_in_bin * np.abs(acc - conf)
    return ece

def test_brier_score_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == 0.0

def test_brier_score_worst():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == 1.0

def test_brier_score_mid():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    assert brier_score(y_true, y_prob) == 0.25

def test_brier_score_stability():
    y_true = np.array([1])
    y_prob = np.array([0.5])
    assert not np.isnan(brier_score(y_true, y_prob))

def test_ece_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert expected_calibration_error(y_true, y_prob_perfect) == 0.0

def test_ece_mid():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(y_true, y_prob) == 0.0

def test_ece_worst():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    ece = expected_calibration_error(y_true, y_prob)
    assert np.isclose(ece, 0.9)

def test_ece_no_nans():
    y_true = np.array([1, 0])
    y_prob = np.array([0.8, 0.2])
    ece = expected_calibration_error(y_true, y_prob)
    assert not np.isnan(ece)

def test_ece_empty_bins():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == 0.0
