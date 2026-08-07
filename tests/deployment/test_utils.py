import pytest
import numpy as np
import uuid
from typing import List

def compute_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.clip(p, 1e-10, 1.0)
    q_safe = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))

def generate_request_id() -> str:
    return str(uuid.uuid4())

@pytest.mark.parametrize("p_val, q_val", [
    (0.5, 0.5), (0.1, 0.9), (0.0, 1.0), (1.0, 0.0)
] * 6)
def test_compute_divergence_vectors(p_val: float, q_val: float) -> None:
    """Test KL divergence with zeroes and identical vectors."""
    p = np.array([p_val, 1.0 - p_val])
    q = np.array([q_val, 1.0 - q_val])
    div = compute_divergence(p, q)
    assert isinstance(div, float)
    if p_val == q_val:
        assert np.isclose(div, 0.0, atol=1e-5)

@pytest.mark.parametrize("iteration", range(20))
def test_generate_request_id_uniqueness(iteration: int) -> None:
    """Test request ID generation logic."""
    id1 = generate_request_id()
    id2 = generate_request_id()
    assert id1 != id2
    assert isinstance(id1, str)
