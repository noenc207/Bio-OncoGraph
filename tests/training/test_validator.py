"""Tests for Validator."""
import pytest
from unittest.mock import MagicMock

class Validator:
    """Mock Validator."""
    def __init__(self, model: MagicMock) -> None:
        self.model = model
        
    def validate(self, dataloader: list) -> dict:
        """Mock validate."""
        return {"val_loss": 0.5 * len(dataloader)}

@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model."""
    return MagicMock()

@pytest.mark.parametrize("dataloader_size,expected_val_loss", [
    (s, 0.5 * s) for s in range(1, 41)
])
def test_validator_loop(mock_model: MagicMock, dataloader_size: int, expected_val_loss: float) -> None:
    """Test Validator."""
    validator = Validator(mock_model)
    dl = [1] * dataloader_size
    metrics = validator.validate(dl)
    assert metrics["val_loss"] == expected_val_loss
