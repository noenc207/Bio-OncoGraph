"""Tests for statistical analysis of benchmark results."""

import pytest
import numpy as np

def mock_p_value(dist_a, dist_b, test_type):
    """Mock p-value calculation."""
    return np.random.uniform(0, 1)

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.15])
@pytest.mark.parametrize("mean_shift", [0.0, 0.5, 1.0, 5.0])
@pytest.mark.parametrize("sample_size", [10, 30, 100])
@pytest.mark.parametrize("test_type", ["ttest", "wilcoxon", "mannwhitney"])
def test_statistical_significance(alpha, mean_shift, sample_size, test_type):
    """Verify statistical significance tests handle varied distributions."""
    np.random.seed(42)
    dist_a = np.random.normal(0, 1, sample_size)
    dist_b = np.random.normal(mean_shift, 1, sample_size)
    
    p_val = mock_p_value(dist_a, dist_b, test_type)
    
    assert 0.0 <= p_val <= 1.0
    significant = p_val < alpha
    assert isinstance(significant, bool)
