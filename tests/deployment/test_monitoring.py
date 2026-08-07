import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

class MonitoringService:
    def track_latency(self, latency: float) -> bool:
        return latency > 0
    def track_throughput(self, req_per_sec: int) -> bool:
        return req_per_sec >= 0
    def track_memory(self, payload_size: int) -> int:
        return payload_size * 2

@pytest.fixture
def monitor() -> MonitoringService:
    return MonitoringService()

@pytest.mark.parametrize("latency, expected", [(float(l), float(l) > 0) for l in np.linspace(-10.0, 100.0, 31)])
def test_track_latency(monitor: MonitoringService, latency: float, expected: bool) -> None:
    """Test tracking latency with varying payload values."""
    assert monitor.track_latency(latency) == expected

@pytest.mark.parametrize("rps", list(range(31)))
def test_track_throughput(monitor: MonitoringService, rps: int) -> None:
    """Test tracking throughput metrics."""
    assert monitor.track_throughput(rps) == (rps >= 0)

@pytest.mark.parametrize("size", list(range(10, 41)))
def test_track_memory(monitor: MonitoringService, size: int) -> None:
    """Test memory tracking logic."""
    assert monitor.track_memory(size) == size * 2
