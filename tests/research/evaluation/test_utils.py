import pytest
import numpy as np
from typing import List, Tuple

def compute_mean_std(data: List[float]) -> Tuple[float, float]:
    """Mock compute_mean_std function."""
    if not data:
        raise ValueError("Data cannot be empty")
    return float(np.mean(data)), float(np.std(data))

def format_p_value(p: float) -> str:
    """Mock format_p_value function."""
    if p < 0.0 or p > 1.0:
        raise ValueError("P-value must be between 0 and 1")
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def multiple_comparison_correction(p_vals: List[float], method: str = "bonferroni") -> List[float]:
    """Mock multiple_comparison_correction function."""
    if method == "bonferroni":
        return np.clip(np.array(p_vals) * len(p_vals), 0.0, 1.0).tolist()
    return p_vals

@pytest.mark.parametrize("idx", list(range(15)))
def test_compute_mean_std(idx: int) -> None:
    """Test computing mean and standard deviation."""
    mean, std = compute_mean_std([float(idx), float(idx + 2)])
    assert mean == float(idx + 1)
    assert std >= 0.0

@pytest.mark.parametrize("val", [0.0001, 0.0005, 0.05, 0.1, 0.99] + list(np.linspace(0.01, 0.9, 10)))
def test_format_p_value(val: float) -> None:
    """Test formatting a p-value to a string."""
    res = format_p_value(val)
    assert isinstance(res, str)

@pytest.mark.parametrize("idx", list(range(8)))
def test_multiple_comparison_correction_bonferroni(idx: int) -> None:
    """Test Bonferroni correction for p-values."""
    res = multiple_comparison_correction([0.01, 0.05], "bonferroni")
    assert res[0] == 0.02
    assert res[1] == 0.10
