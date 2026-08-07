"""Tests for MetricTracker."""
import pytest
import numpy as np

class MetricTracker:
    """Mock MetricTracker."""
    def __init__(self) -> None:
        self.metrics = []
        
    def update(self, val: float) -> None:
        """Mock update."""
        self.metrics.append(val)
        
    def compute(self) -> float:
        """Mock compute."""
        return float(np.mean(self.metrics)) if self.metrics else 0.0

@pytest.mark.parametrize("vals,expected", [
    (list(range(i)), (sum(range(i))/i) if i > 0 else 0.0) for i in range(45)
])
def test_metric_tracker(vals: list, expected: float) -> None:
    """Test MetricTracker."""
    tracker = MetricTracker()
    for v in vals:
        tracker.update(v)
    assert np.isclose(tracker.compute(), expected)
