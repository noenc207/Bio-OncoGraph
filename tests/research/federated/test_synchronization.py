"""Tests for Synchronization."""
import pytest
from typing import Dict, List

class FederatedSynchronizer:
    """Mock Synchronizer."""
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.start_times: Dict[str, float] = {}
        
    def start_client(self, cid: str, t: float) -> None:
        """Record start time."""
        self.start_times[cid] = t
        
    def detect_stragglers(self, current_time: float) -> List[str]:
        """Detect straggling clients based on timeout."""
        return [cid for cid, t in self.start_times.items() if current_time - t > self.timeout]

@pytest.mark.parametrize("timeout", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("num_clients", [5, 10, 20])
@pytest.mark.parametrize("straggler_ratio", [0.0, 0.2, 0.5, 1.0])
def test_detect_stragglers(timeout: float, num_clients: int, straggler_ratio: float) -> None:
    """Test straggler detection."""
    sync = FederatedSynchronizer(timeout)
    current_time = 100.0
    
    num_stragglers = int(num_clients * straggler_ratio)
    
    for i in range(num_clients):
        cid = f"client_{i}"
        if i < num_stragglers:
            # Started before timeout threshold
            sync.start_client(cid, current_time - timeout - 1.0)
        else:
            # Started within timeout threshold
            sync.start_client(cid, current_time - timeout + 1.0)
            
    stragglers = sync.detect_stragglers(current_time)
    assert len(stragglers) == num_stragglers

@pytest.mark.parametrize("timeout", [2.0, 4.0])
@pytest.mark.parametrize("clients", [["a", "b"], ["x", "y", "z"]])
def test_no_stragglers(timeout: float, clients: List[str]) -> None:
    """Test no stragglers."""
    sync = FederatedSynchronizer(timeout)
    for c in clients:
        sync.start_client(c, 0.0)
    assert len(sync.detect_stragglers(0.0)) == 0
