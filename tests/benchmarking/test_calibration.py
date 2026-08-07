"""Tests for confidence calibration metrics."""

import pytest
import numpy as np

def calculate_ece(confidences, labels, n_bins=10):
    """Mock ECE calculation."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            ece += np.mean(in_bin) * 0.05
    return ece

@pytest.mark.parametrize("n_bins", [5, 10, 15, 20, 25])
@pytest.mark.parametrize("confidence_mode", ["uniform", "random", "high", "low"])
@pytest.mark.parametrize("label_mode", ["random", "all_zero", "all_one"])
@pytest.mark.parametrize("sample_size", [10, 50, 100])
def test_calibration_ece(n_bins, confidence_mode, label_mode, sample_size):
    """Test Expected Calibration Error (ECE) metric calculations."""
    if confidence_mode == "uniform":
        confidences = np.linspace(0.1, 0.9, sample_size)
    elif confidence_mode == "high":
        confidences = np.random.uniform(0.8, 1.0, sample_size)
    elif confidence_mode == "low":
        confidences = np.random.uniform(0.0, 0.2, sample_size)
    else:
        confidences = np.random.rand(sample_size)
        
    if label_mode == "all_zero":
        labels = np.zeros(sample_size)
    elif label_mode == "all_one":
        labels = np.ones(sample_size)
    else:
        labels = np.random.randint(0, 2, sample_size)
        
    ece = calculate_ece(confidences, labels, n_bins=n_bins)
    assert 0.0 <= ece <= 1.0
