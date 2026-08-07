import pytest
from typing import List

class RoundScheduler:
    def sample_clients(self, clients: List[str], num_samples: int, seed: int) -> List[str]:
        import random
        random.seed(seed)
        if not clients or num_samples == 0:
            return []
        return random.sample(clients, min(len(clients), num_samples))
    def handle_timeout(self, client_id: str) -> bool:
        return True

@pytest.mark.parametrize("total_clients", [10, 50, 100])
@pytest.mark.parametrize("num_samples", [0, 5, 20, 150])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_scheduler_sample_clients(total_clients: int, num_samples: int, seed: int) -> None:
    """Test uniform client sampling."""
    scheduler = RoundScheduler()
    clients = [f"client_{i}" for i in range(total_clients)]
    sampled = scheduler.sample_clients(clients, num_samples, seed)
    expected_len = min(total_clients, num_samples) if num_samples > 0 else 0
    assert len(sampled) == expected_len
    
    # Test determinism
    sampled_again = scheduler.sample_clients(clients, num_samples, seed)
    assert sampled == sampled_again

@pytest.mark.parametrize("client_id", [f"client_{i}" for i in range(10)])
def test_scheduler_handle_timeout(client_id: str) -> None:
    """Test timeout handling logic."""
    scheduler = RoundScheduler()
    assert scheduler.handle_timeout(client_id) is True
