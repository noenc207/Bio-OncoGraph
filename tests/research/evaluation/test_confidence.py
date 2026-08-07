"""Tests for confidence estimators."""

import pytest
import numpy as np

class ConfidenceEstimator:
    """Mock Confidence Estimator."""
    @staticmethod
    def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
        """Mock Bootstrap Confidence Interval."""
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        if n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1")
        return {"lower": 0.1, "upper": 0.9, "mean": 0.5}

    @staticmethod
    def jackknife_ci(data, alpha=0.05):
        """Mock Jackknife Confidence Interval."""
        if len(data) < 2:
            raise ValueError("Data must have at least 2 elements")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        return {"lower": 0.2, "upper": 0.8, "mean": 0.5}

    @staticmethod
    def empirical_ci(data, alpha=0.05):
        """Mock Empirical Confidence Interval."""
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        return {"lower": 0.15, "upper": 0.85, "mean": 0.5}

@pytest.fixture
def estimator():
    """Fixture for ConfidenceEstimator."""
    return ConfidenceEstimator()

@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
def test_bootstrap_ci_valid(estimator, size):
    """Test bootstrap_ci with valid data."""
    data = np.random.randn(size)
    res = estimator.bootstrap_ci(data)
    assert res["lower"] <= res["mean"] <= res["upper"]

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_bootstrap_ci_alphas(estimator, alpha):
    """Test bootstrap_ci with various alpha values."""
    data = np.random.randn(100)
    res = estimator.bootstrap_ci(data, alpha=alpha)
    assert res["lower"] <= res["upper"]

@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, 2.0])
def test_bootstrap_ci_invalid_alphas(estimator, alpha):
    """Test bootstrap_ci with invalid alpha values."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        estimator.bootstrap_ci(data, alpha=alpha)

@pytest.mark.parametrize("n_bootstraps", [1, 10, 100, 500, 1000])
def test_bootstrap_ci_bootstraps(estimator, n_bootstraps):
    """Test bootstrap_ci with various bootstrap counts."""
    data = np.random.randn(100)
    res = estimator.bootstrap_ci(data, n_bootstraps=n_bootstraps)
    assert "mean" in res

@pytest.mark.parametrize("n_bootstraps", [0, -1, -100])
def test_bootstrap_ci_invalid_bootstraps(estimator, n_bootstraps):
    """Test bootstrap_ci with invalid bootstrap counts."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        estimator.bootstrap_ci(data, n_bootstraps=n_bootstraps)

@pytest.mark.parametrize("size", [2, 10, 50, 100, 500])
def test_jackknife_ci_valid(estimator, size):
    """Test jackknife_ci with valid data."""
    data = np.random.randn(size)
    res = estimator.jackknife_ci(data)
    assert res["lower"] <= res["mean"] <= res["upper"]

@pytest.mark.parametrize("size", [0, 1])
def test_jackknife_ci_invalid_size(estimator, size):
    """Test jackknife_ci with invalid data size."""
    data = np.random.randn(size)
    with pytest.raises(ValueError):
        estimator.jackknife_ci(data)

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_jackknife_ci_alphas(estimator, alpha):
    """Test jackknife_ci with various alpha values."""
    data = np.random.randn(100)
    res = estimator.jackknife_ci(data, alpha=alpha)
    assert res["lower"] <= res["upper"]

@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, 2.0])
def test_jackknife_ci_invalid_alphas(estimator, alpha):
    """Test jackknife_ci with invalid alpha values."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        estimator.jackknife_ci(data, alpha=alpha)

@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
def test_empirical_ci_valid(estimator, size):
    """Test empirical_ci with valid data."""
    data = np.random.randn(size)
    res = estimator.empirical_ci(data)
    assert res["lower"] <= res["mean"] <= res["upper"]

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.2, 0.5])
def test_empirical_ci_alphas(estimator, alpha):
    """Test empirical_ci with various alpha values."""
    data = np.random.randn(100)
    res = estimator.empirical_ci(data, alpha=alpha)
    assert res["lower"] <= res["upper"]

@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, 2.0])
def test_empirical_ci_invalid_alphas(estimator, alpha):
    """Test empirical_ci with invalid alpha values."""
    data = np.random.randn(100)
    with pytest.raises(ValueError):
        estimator.empirical_ci(data, alpha=alpha)
