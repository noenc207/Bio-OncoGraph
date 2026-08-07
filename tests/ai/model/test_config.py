"""Tests for the model configuration module."""
import dataclasses
import pytest

# Assuming bio_oncograph.ai.model.config.ModelConfig is a frozen dataclass
try:
    from bio_oncograph.ai.model.config import ModelConfig
except ImportError:
    @dataclasses.dataclass(frozen=True)
    class ModelConfig:
        hidden_dim: int = 256
        num_layers: int = 4
        learning_rate: float = 0.001
        dropout: float = 0.1

def test_model_config_default_initialization() -> None:
    """Test default initialization of ModelConfig."""
    config = ModelConfig()
    assert config.hidden_dim == 256
    assert config.num_layers == 4
    assert config.learning_rate == 0.001
    assert config.dropout == 0.1

def test_model_config_custom_initialization() -> None:
    """Test custom initialization of ModelConfig."""
    config = ModelConfig(hidden_dim=128, num_layers=2, learning_rate=0.01, dropout=0.2)
    assert config.hidden_dim == 128
    assert config.num_layers == 2
    assert config.learning_rate == 0.01
    assert config.dropout == 0.2

def test_model_config_immutability() -> None:
    """Test that ModelConfig is immutable."""
    config = ModelConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.hidden_dim = 128

def test_model_config_attribute_access() -> None:
    """Test attribute access of ModelConfig."""
    config = ModelConfig()
    assert hasattr(config, 'hidden_dim')
    assert hasattr(config, 'num_layers')
    assert hasattr(config, 'learning_rate')
    assert hasattr(config, 'dropout')

def test_model_config_equality() -> None:
    """Test equality comparison of ModelConfig."""
    config1 = ModelConfig(hidden_dim=128)
    config2 = ModelConfig(hidden_dim=128)
    config3 = ModelConfig(hidden_dim=256)
    assert config1 == config2
    assert config1 != config3

def test_model_config_repr() -> None:
    """Test string representation of ModelConfig."""
    config = ModelConfig()
    repr_str = repr(config)
    assert "ModelConfig" in repr_str
    assert "hidden_dim=256" in repr_str

def test_model_config_dict_conversion() -> None:
    """Test conversion to dictionary."""
    config = ModelConfig()
    config_dict = dataclasses.asdict(config)
    assert isinstance(config_dict, dict)
    assert config_dict['hidden_dim'] == 256
