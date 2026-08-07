"""Tests for RegressionTester and backward compatibility."""

import pytest
import numpy as np
from typing import Any, Dict, List, Optional


class RegressionTester:
    """Mock RegressionTester for integration testing."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self.history: List[Dict[str, Any]] = []

    def check_api_compatibility(self, old_api: Dict[str, Any], new_api: Dict[str, Any]) -> bool:
        """Check API compatibility between versions."""
        if not self.strict:
            return True
        for key in old_api:
            if key not in new_api:
                return False
        return True

    def check_backward_compatibility(self, flags: Dict[str, bool]) -> bool:
        """Check if all flags maintain backward compatibility."""
        return all(flags.values())

    def simulate_run(self, data: np.ndarray) -> np.ndarray:
        """Simulate a regression test run."""
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        return data * 2

    def record_run(self, result: float) -> None:
        """Record run result."""
        self.history.append({"result": result})

    def get_history(self) -> List[Dict[str, Any]]:
        """Get run history."""
        return self.history

    def clear_history(self) -> None:
        """Clear history."""
        self.history.clear()


@pytest.fixture
def tester() -> RegressionTester:
    return RegressionTester(strict=True)


def test_regression_tester_init(tester: RegressionTester) -> None:
    assert tester.strict is True


def test_regression_tester_init_non_strict() -> None:
    tester = RegressionTester(strict=False)
    assert tester.strict is False


def test_api_compatibility_identical(tester: RegressionTester) -> None:
    api = {"func1": True, "func2": True}
    assert tester.check_api_compatibility(api, api) is True


def test_api_compatibility_missing_key(tester: RegressionTester) -> None:
    old_api = {"func1": True, "func2": True}
    new_api = {"func1": True}
    assert tester.check_api_compatibility(old_api, new_api) is False


def test_api_compatibility_extra_key(tester: RegressionTester) -> None:
    old_api = {"func1": True}
    new_api = {"func1": True, "func2": True}
    assert tester.check_api_compatibility(old_api, new_api) is True


def test_api_compatibility_non_strict() -> None:
    tester = RegressionTester(strict=False)
    old_api = {"func1": True, "func2": True}
    new_api = {"func1": True}
    assert tester.check_api_compatibility(old_api, new_api) is True


def test_backward_compatibility_all_true(tester: RegressionTester) -> None:
    flags = {"flag1": True, "flag2": True}
    assert tester.check_backward_compatibility(flags) is True


def test_backward_compatibility_one_false(tester: RegressionTester) -> None:
    flags = {"flag1": True, "flag2": False}
    assert tester.check_backward_compatibility(flags) is False


def test_backward_compatibility_empty(tester: RegressionTester) -> None:
    assert tester.check_backward_compatibility({}) is True


def test_simulate_run_valid(tester: RegressionTester) -> None:
    data = np.array([1, 2, 3])
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.array([2, 4, 6]))


def test_simulate_run_empty(tester: RegressionTester) -> None:
    data = np.array([])
    with pytest.raises(ValueError):
        tester.simulate_run(data)


def test_simulate_run_zeros(tester: RegressionTester) -> None:
    data = np.zeros(5)
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.zeros(5))


def test_simulate_run_negative(tester: RegressionTester) -> None:
    data = np.array([-1, -2, -3])
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.array([-2, -4, -6]))


def test_simulate_run_floats(tester: RegressionTester) -> None:
    data = np.array([1.5, 2.5])
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.array([3.0, 5.0]))


def test_record_run(tester: RegressionTester) -> None:
    tester.record_run(1.0)
    assert len(tester.get_history()) == 1
    assert tester.get_history()[0]["result"] == 1.0


def test_record_multiple_runs(tester: RegressionTester) -> None:
    tester.record_run(1.0)
    tester.record_run(2.0)
    assert len(tester.get_history()) == 2


def test_clear_history(tester: RegressionTester) -> None:
    tester.record_run(1.0)
    tester.clear_history()
    assert len(tester.get_history()) == 0


def test_get_history_empty(tester: RegressionTester) -> None:
    assert len(tester.get_history()) == 0


def test_simulate_run_large_data(tester: RegressionTester) -> None:
    data = np.ones(1000)
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.full(1000, 2))


def test_simulate_run_2d_array(tester: RegressionTester) -> None:
    data = np.array([[1, 2], [3, 4]])
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.array([[2, 4], [6, 8]]))


def test_simulate_run_3d_array(tester: RegressionTester) -> None:
    data = np.ones((2, 2, 2))
    result = tester.simulate_run(data)
    np.testing.assert_array_equal(result, np.full((2, 2, 2), 2))


def test_api_compatibility_empty_old(tester: RegressionTester) -> None:
    assert tester.check_api_compatibility({}, {"func1": True}) is True


def test_api_compatibility_empty_new(tester: RegressionTester) -> None:
    assert tester.check_api_compatibility({"func1": True}, {}) is False


def test_api_compatibility_both_empty(tester: RegressionTester) -> None:
    assert tester.check_api_compatibility({}, {}) is True


def test_backward_compatibility_all_false(tester: RegressionTester) -> None:
    flags = {"flag1": False, "flag2": False}
    assert tester.check_backward_compatibility(flags) is False


def test_record_run_zero(tester: RegressionTester) -> None:
    tester.record_run(0.0)
    assert tester.get_history()[0]["result"] == 0.0


def test_record_run_negative(tester: RegressionTester) -> None:
    tester.record_run(-5.5)
    assert tester.get_history()[0]["result"] == -5.5
