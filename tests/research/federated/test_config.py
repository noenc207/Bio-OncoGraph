"""Tests for FederatedConfig."""
import pytest
from dataclasses import dataclass

@dataclass(frozen=True)
class FederatedConfig:
    """Configuration for Federated Learning."""
    num_rounds: int = 10
    fraction_fit: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    def __post_init__(self) -> None:
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be > 0")
        if not (0.0 < self.fraction_fit <= 1.0):
            raise ValueError("fraction_fit must be in (0, 1]")

@pytest.mark.parametrize("num_rounds", [1, 5, 10, 100, 500])
@pytest.mark.parametrize("fraction_fit", [0.1, 0.5, 1.0])
def test_config_creation_valid(num_rounds: int, fraction_fit: float) -> None:
    """Test valid configuration creation."""
    config = FederatedConfig(num_rounds=num_rounds, fraction_fit=fraction_fit)
    assert config.num_rounds == num_rounds
    assert config.fraction_fit == fraction_fit
    assert config.min_fit_clients == 2

@pytest.mark.parametrize("num_rounds", [0, -1, -5])
def test_config_invalid_num_rounds(num_rounds: int) -> None:
    """Test invalid num_rounds."""
    with pytest.raises(ValueError):
        FederatedConfig(num_rounds=num_rounds)

@pytest.mark.parametrize("fraction_fit", [0.0, -0.1, 1.1, 2.0])
def test_config_invalid_fraction_fit(fraction_fit: float) -> None:
    """Test invalid fraction_fit."""
    with pytest.raises(ValueError):
        FederatedConfig(fraction_fit=fraction_fit)

@pytest.mark.parametrize("min_fit", [1, 2, 5])
@pytest.mark.parametrize("min_eval", [1, 2, 5])
def test_config_client_counts(min_fit: int, min_eval: int) -> None:
    """Test client counts."""
    config = FederatedConfig(min_fit_clients=min_fit, min_evaluate_clients=min_eval)
    assert config.min_fit_clients == min_fit
    assert config.min_evaluate_clients == min_eval
