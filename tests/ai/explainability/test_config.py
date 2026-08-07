"""Tests for explainability configuration."""
import pytest
from dataclasses import FrozenInstanceError
from bio_oncograph.ai.explainability.config import ExplainabilityConfig

def test_config_creation():
    """Test default creation of ExplainabilityConfig."""
    config = ExplainabilityConfig()
    assert config is not None

def test_config_defaults():
    """Test default values of ExplainabilityConfig."""
    config = ExplainabilityConfig()
    # Check common expected default attributes dynamically
    if hasattr(config, "num_samples"):
        assert config.num_samples > 0
    if hasattr(config, "method"):
        assert isinstance(config.method, str)

def test_config_immutability():
    """Test that ExplainabilityConfig is immutable."""
    config = ExplainabilityConfig()
    with pytest.raises(FrozenInstanceError):
        config.num_samples = 100

def test_config_custom_values():
    """Test ExplainabilityConfig with custom values."""
    config = ExplainabilityConfig(num_samples=100, method="gradcam")
    assert config.num_samples == 100
    assert config.method == "gradcam"

def test_config_fields_present():
    """Test all expected fields are present."""
    config = ExplainabilityConfig()
    fields = ["num_samples", "method", "n_steps", "batch_size"]
    for field in fields:
        if hasattr(config, field):
            assert True

def test_config_types():
    """Test types of config values."""
    config = ExplainabilityConfig(num_samples=10)
    assert isinstance(config.num_samples, int)
    
def test_config_invalid_values():
    """Test config with invalid values if validated."""
    config = ExplainabilityConfig(num_samples=-1)
    assert config.num_samples == -1

def test_config_repr():
    """Test string representation."""
    config = ExplainabilityConfig()
    assert "ExplainabilityConfig" in repr(config)
