import pytest
import numpy as np
from typing import Dict, Any

class SyncBackend:
    def send(self, data: Dict[str, Any]) -> bool: return True
    def receive(self) -> Dict[str, Any]: return {"status": "ok"}

class AsyncBackend:
    async def send(self, data: Dict[str, Any]) -> bool: return True
    async def receive(self) -> Dict[str, Any]: return {"status": "ok"}

class RetryPolicy:
    def __init__(self, retries: int = 3): self.retries = retries
    def should_retry(self, attempt: int) -> bool: return attempt < self.retries

@pytest.mark.parametrize("data_size", [10, 100, 1000])
@pytest.mark.parametrize("include_metadata", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_sync_backend_send(data_size: int, include_metadata: bool, seed: int) -> None:
    """Test synchronous backend sending."""
    np.random.seed(seed)
    backend = SyncBackend()
    data = {"payload": np.random.rand(data_size)}
    if include_metadata:
        data["metadata"] = {"id": 1}
    assert backend.send(data) is True

@pytest.mark.parametrize("attempt", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("max_retries", [1, 3, 5])
def test_retry_policy(attempt: int, max_retries: int) -> None:
    """Test retry policy logic."""
    policy = RetryPolicy(retries=max_retries)
    expected = attempt < max_retries
    assert policy.should_retry(attempt) == expected

@pytest.mark.asyncio
@pytest.mark.parametrize("data_size", [50, 150, 250])
@pytest.mark.parametrize("seed", [10, 11, 12, 13])
async def test_async_backend_send(data_size: int, seed: int) -> None:
    """Test asynchronous backend sending."""
    np.random.seed(seed)
    backend = AsyncBackend()
    data = {"payload": np.random.rand(data_size)}
    result = await backend.send(data)
    assert result is True
