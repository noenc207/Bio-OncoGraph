import pytest
import numpy as np
from typing import List

class MetricsCollector:
    def aggregate(self, metrics: List[float], method: str) -> float:
        if not metrics:
            return 0.0
        if method == "mean":
            return float(np.mean(metrics))
        if method == "max":
            return float(np.max(metrics))
        if method == "min":
            return float(np.min(metrics))
        return float(np.sum(metrics))

@pytest.fixture
def collector() -> MetricsCollector:
    return MetricsCollector()

@pytest.mark.parametrize("metrics, method, expected", [
    ([1.0, 2.0, 3.0], "mean", 2.0),
    ([1.0, 2.0, 3.0], "max", 3.0),
    ([1.0, 2.0, 3.0], "min", 1.0),
    ([1.0, 2.0, 3.0], "sum", 6.0),
    ([], "mean", 0.0)
] * 10)
def test_aggregate_metrics(collector: MetricsCollector, metrics: List[float], method: str, expected: float) -> None:
    """Test aggregations of metrics lists for different methods."""
    assert collector.aggregate(metrics, method) == expected

@pytest.mark.parametrize("size", range(1, 21))
def test_aggregate_dynamic_metrics(collector: MetricsCollector, size: int) -> None:
    """Test collector on dynamic metric lists."""
    metrics = [float(x) for x in range(size)]
    assert collector.aggregate(metrics, "max") == float(size - 1)
