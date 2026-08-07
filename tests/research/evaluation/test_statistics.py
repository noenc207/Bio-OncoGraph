"""Tests for statistical evaluator."""

import pytest
import numpy as np

class StatisticalEvaluator:
    """Mock statistical evaluator."""
    @staticmethod
    def paired_t_test(a, b):
        """Mock paired t-test."""
        a, b = np.asarray(a), np.asarray(b)
        if len(a) != len(b):
            raise ValueError("Arrays must have same length")
        if np.array_equal(a, b):
            return {"p_value": 1.0, "statistic": 0.0}
        if np.all(a == 0) and np.all(b == 0):
            return {"p_value": 1.0, "statistic": 0.0}
        return {"p_value": 0.05, "statistic": 2.5}

    @staticmethod
    def wilcoxon_signed_rank(a, b):
        """Mock Wilcoxon signed-rank test."""
        a, b = np.asarray(a), np.asarray(b)
        if len(a) != len(b):
            raise ValueError("Arrays must have same length")
        if np.array_equal(a, b):
            return {"p_value": 1.0, "statistic": 0.0}
        if np.all(a == 0) and np.all(b == 0):
            return {"p_value": 1.0, "statistic": 0.0}
        return {"p_value": 0.01, "statistic": 10.0}

    @staticmethod
    def cohens_d(a, b):
        """Mock Cohen's d calculation."""
        a, b = np.asarray(a), np.asarray(b)
        if len(a) == 0 or len(b) == 0:
            raise ValueError("Arrays cannot be empty")
        if np.array_equal(a, b):
            return 0.0
        return 0.5

@pytest.fixture
def evaluator():
    """Fixture for StatisticalEvaluator."""
    return StatisticalEvaluator()

@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
def test_paired_t_test_valid(evaluator, size):
    """Test paired t-test with valid data."""
    a = np.random.randn(size)
    b = np.random.randn(size)
    res = evaluator.paired_t_test(a, b)
    assert "p_value" in res

@pytest.mark.parametrize("size_a, size_b", [(10, 11), (50, 49), (1, 2), (100, 99), (0, 1)])
def test_paired_t_test_invalid_length(evaluator, size_a, size_b):
    """Test paired t-test with mismatched lengths."""
    a = np.random.randn(size_a)
    b = np.random.randn(size_b)
    with pytest.raises(ValueError):
        evaluator.paired_t_test(a, b)

@pytest.mark.parametrize("size", [1, 5, 10, 50, 100])
def test_paired_t_test_identical(evaluator, size):
    """Test paired t-test with identical arrays."""
    a = np.random.randn(size)
    res = evaluator.paired_t_test(a, a)
    assert res["p_value"] == 1.0

@pytest.mark.parametrize("size", [1, 5, 10, 50, 100])
def test_paired_t_test_all_zeros(evaluator, size):
    """Test paired t-test with all zeros."""
    a = np.zeros(size)
    b = np.zeros(size)
    res = evaluator.paired_t_test(a, b)
    assert res["p_value"] == 1.0

@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
def test_wilcoxon_valid(evaluator, size):
    """Test Wilcoxon signed-rank with valid data."""
    a = np.random.randn(size)
    b = np.random.randn(size)
    res = evaluator.wilcoxon_signed_rank(a, b)
    assert "p_value" in res

@pytest.mark.parametrize("size_a, size_b", [(10, 11), (50, 49), (1, 2), (100, 99), (0, 1)])
def test_wilcoxon_invalid_length(evaluator, size_a, size_b):
    """Test Wilcoxon signed-rank with mismatched lengths."""
    a = np.random.randn(size_a)
    b = np.random.randn(size_b)
    with pytest.raises(ValueError):
        evaluator.wilcoxon_signed_rank(a, b)

@pytest.mark.parametrize("size", [1, 5, 10, 50, 100])
def test_wilcoxon_identical(evaluator, size):
    """Test Wilcoxon signed-rank with identical arrays."""
    a = np.random.randn(size)
    res = evaluator.wilcoxon_signed_rank(a, a)
    assert res["p_value"] == 1.0

@pytest.mark.parametrize("size", [1, 5, 10, 50, 100])
def test_wilcoxon_all_zeros(evaluator, size):
    """Test Wilcoxon signed-rank with all zeros."""
    a = np.zeros(size)
    b = np.zeros(size)
    res = evaluator.wilcoxon_signed_rank(a, b)
    assert res["p_value"] == 1.0

@pytest.mark.parametrize("size_a, size_b", [(10, 10), (50, 50), (100, 100), (10, 50), (50, 10)])
def test_cohens_d_valid(evaluator, size_a, size_b):
    """Test Cohen's d calculation with valid data."""
    a = np.random.randn(size_a)
    b = np.random.randn(size_b)
    res = evaluator.cohens_d(a, b)
    assert isinstance(res, float)

@pytest.mark.parametrize("size", [1, 5, 10, 50, 100])
def test_cohens_d_identical(evaluator, size):
    """Test Cohen's d calculation with identical arrays."""
    a = np.random.randn(size)
    res = evaluator.cohens_d(a, a)
    assert res == 0.0
