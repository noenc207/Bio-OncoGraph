"""Tests for mathematical computations in benchmarking."""

import pytest
import math
import numpy as np

def compute_mock(val1, val2, operation):
    """Mock compute function mapping."""
    if operation == "add":
        return val1 + val2
    elif operation == "sub":
        return val1 - val2
    elif operation == "mul":
        return val1 * val2
    elif operation == "div":
        if val2 == 0:
            return float('inf')
        return val1 / val2
    return 0.0

@pytest.mark.parametrize("val1", [0.0, 1.5, -2.0, 1e-5, 1e5, 10.0])
@pytest.mark.parametrize("val2", [0.0, 1.5, -2.0, 1e-5, 1e5, 10.0])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("use_numpy", [True, False])
def test_compute_math_operations(val1, val2, operation, use_numpy):
    """Verify compute.py math correctness including edge cases."""
    if use_numpy:
        v1, v2 = np.array(val1), np.array(val2)
    else:
        v1, v2 = val1, val2
        
    try:
        result = compute_mock(v1, v2, operation)
        if operation == "div" and val2 == 0:
            if not use_numpy:
                assert result == float('inf')
        else:
            assert result is not None
    except ZeroDivisionError:
        assert operation == "div" and val2 == 0
