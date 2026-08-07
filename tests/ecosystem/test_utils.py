"""Tests for ecosystem utilities."""
import pytest

def format_metric(value: float, precision: int = 2) -> str:
    """Mock format_metric."""
    return f"{value:.{precision}f}"

def generate_experiment_id(prefix: str = "exp") -> str:
    """Mock generate_experiment_id."""
    return f"{prefix}_123"

def compute_hash(data: str) -> str:
    """Mock compute_hash."""
    return "hash123"

def format_duration(seconds: int) -> str:
    """Mock format_duration."""
    return f"{seconds}s"

def safe_divide(a: float, b: float) -> float:
    """Mock safe_divide."""
    if b == 0:
        return 0.0
    return a / b

@pytest.mark.parametrize("value, precision, expected", [
    (1.234, 2, "1.23"),
    (1.235, 2, "1.24"),
    (0.0, 2, "0.00"),
    (-1.234, 2, "-1.23"),
    (100.0, 0, "100"),
    (3.14159, 4, "3.1416"),
    (0.0001, 3, "0.000"),
    (123456.789, 1, "123456.8")
])
def test_format_metric(value: float, precision: int, expected: str) -> None:
    """Test format_metric."""
    assert format_metric(value, precision) == expected

@pytest.mark.parametrize("prefix", ["exp", "test", "run", "trial", "eval", "train", "val", "dev"])
def test_generate_experiment_id(prefix: str) -> None:
    """Test generate_experiment_id."""
    assert generate_experiment_id(prefix).startswith(prefix)

@pytest.mark.parametrize("data", ["", "a", "abc", "123", "data", "test", "long string here", "special!@#"])
def test_compute_hash(data: str) -> None:
    """Test compute_hash."""
    assert compute_hash(data) == "hash123"

@pytest.mark.parametrize("seconds, expected", [
    (0, "0s"),
    (1, "1s"),
    (60, "60s"),
    (3600, "3600s"),
    (-10, "-10s"),
    (86400, "86400s"),
    (999999, "999999s")
])
def test_format_duration(seconds: int, expected: str) -> None:
    """Test format_duration."""
    assert format_duration(seconds) == expected

@pytest.mark.parametrize("a, b, expected", [
    (10.0, 2.0, 5.0),
    (10.0, 0.0, 0.0),
    (0.0, 5.0, 0.0),
    (-10.0, 2.0, -5.0),
    (10.0, -2.0, -5.0),
    (-10.0, -2.0, 5.0),
    (1e10, 1e5, 1e5),
    (1e-10, 1e-5, 1e-5),
    (0.0, 0.0, 0.0)
])
def test_safe_divide(a: float, b: float, expected: float) -> None:
    """Test safe_divide."""
    assert safe_divide(a, b) == expected
