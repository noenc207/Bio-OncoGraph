import pytest
import numpy as np
from typing import Dict, Any

class FederatedValidator:
    def validate_client(self, client_data: Dict[str, Any]) -> bool:
        return "id" in client_data
    def validate_aggregation(self, updates: np.ndarray) -> bool:
        if np.isnan(updates).any():
            return False
        return True
    def validate_round(self, round_num: int) -> bool:
        return round_num > 0

@pytest.mark.parametrize("client_id", [1, 2, 3, None])
@pytest.mark.parametrize("has_data", [True, False])
def test_validator_validate_client(client_id: Any, has_data: bool) -> None:
    """Test validation of client data."""
    validator = FederatedValidator()
    data: Dict[str, Any] = {}
    if client_id is not None:
        data["id"] = client_id
    if has_data:
        data["payload"] = "data"
    expected = "id" in data
    assert validator.validate_client(data) == expected

@pytest.mark.parametrize("shape", [(10,), (5, 5), (100, 10), (50, 50)])
@pytest.mark.parametrize("contains_nan", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_validator_validate_aggregation(shape: tuple, contains_nan: bool, seed: int) -> None:
    """Test aggregation validation including NaN checks."""
    np.random.seed(seed)
    validator = FederatedValidator()
    updates = np.random.rand(*shape)
    if contains_nan:
        updates.flat[0] = np.nan
    expected = not contains_nan
    assert validator.validate_aggregation(updates) == expected

@pytest.mark.parametrize("round_num", [-5, 0, 1, 10, 100])
def test_validator_validate_round(round_num: int) -> None:
    """Test validation of round numbers."""
    validator = FederatedValidator()
    assert validator.validate_round(round_num) == (round_num > 0)
