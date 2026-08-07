"""Tests for foundation model configurations."""
import pytest
from typing import Any

class FoundationModelConfig:
    """Dummy FoundationModelConfig for testing purposes."""
    def __init__(self, model_name: str = "default", max_length: int = 512, temperature: float = 0.7):
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Invalid model_name")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("Invalid max_length")
        if not isinstance(temperature, float) or not 0 <= temperature <= 1:
            raise ValueError("Invalid temperature")
        self._model_name = model_name
        self._max_length = max_length
        self._temperature = temperature
        
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name
        
    @property
    def max_length(self) -> int:
        """Get maximum sequence length."""
        return self._max_length
        
    @property
    def temperature(self) -> float:
        """Get generation temperature."""
        return self._temperature


@pytest.mark.parametrize("model_name", ["gpt-4", "llama-2", "bert-base", "bio-bert", "clinical-longformer"])
@pytest.mark.parametrize("max_length", [128, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("temperature", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_config_initialization(model_name: str, max_length: int, temperature: float) -> None:
    """Test valid initialization of FoundationModelConfig."""
    config = FoundationModelConfig(model_name=model_name, max_length=max_length, temperature=temperature)
    assert config.model_name == model_name
    assert config.max_length == max_length
    assert config.temperature == temperature

@pytest.mark.parametrize("invalid_name", ["", None, 123, [], {}, set()])
def test_config_invalid_model_name(invalid_name: Any) -> None:
    """Test initialization with invalid model names."""
    with pytest.raises((ValueError, TypeError)):
        FoundationModelConfig(model_name=invalid_name)

@pytest.mark.parametrize("invalid_length", [0, -1, -100, 1.5, "512", None])
def test_config_invalid_max_length(invalid_length: Any) -> None:
    """Test initialization with invalid max lengths."""
    with pytest.raises((ValueError, TypeError)):
        FoundationModelConfig(max_length=invalid_length)

@pytest.mark.parametrize("invalid_temp", [-0.1, 1.1, -5.0, 5.0, "0.5", int(1)])
def test_config_invalid_temperature(invalid_temp: Any) -> None:
    """Test initialization with invalid temperatures."""
    with pytest.raises((ValueError, TypeError)):
        FoundationModelConfig(temperature=invalid_temp)
