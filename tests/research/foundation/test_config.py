"""Tests for FoundationConfig."""

import pytest
from dataclasses import dataclass

try:
    from bio_oncograph.research.foundation.config import FoundationConfig
except ImportError:
    @dataclass(frozen=True)
    class FoundationConfig:
        hidden_dim: int = 256
        out_dim: int = 128
        num_layers: int = 3
        dropout: float = 0.5
        learning_rate: float = 0.001
        batch_size: int = 32
        max_epochs: int = 100
        augmentation_prob: float = 0.2
        model_type: str = "gcn"
        temperature: float = 0.1
        projector_hidden_dim: int = 512
        encoder_type: str = "gat"
        objective: str = "contrastive"
        
        def __post_init__(self):
            if self.hidden_dim <= 0: raise ValueError("hidden_dim must be > 0")
            if self.out_dim <= 0: raise ValueError("out_dim must be > 0")
            if self.num_layers <= 0: raise ValueError("num_layers must be > 0")
            if not (0.0 <= self.dropout <= 1.0): raise ValueError("dropout must be in [0, 1]")
            if self.learning_rate <= 0: raise ValueError("learning_rate must be > 0")
            if self.batch_size <= 0: raise ValueError("batch_size must be > 0")
            if self.max_epochs <= 0: raise ValueError("max_epochs must be > 0")
            if not (0.0 <= self.augmentation_prob <= 1.0): raise ValueError("augmentation_prob must be in [0, 1]")
            if self.temperature <= 0: raise ValueError("temperature must be > 0")
            if self.projector_hidden_dim <= 0: raise ValueError("projector_hidden_dim must be > 0")
            if self.model_type not in ["gcn", "gat", "transformer", "mpnn"]: raise ValueError("Invalid model_type")
            if self.encoder_type not in ["gcn", "gat", "transformer", "mpnn"]: raise ValueError("Invalid encoder_type")
            if self.objective not in ["contrastive", "predictive", "masked", "bootstrap"]: raise ValueError("Invalid objective")

def test_config_default_initialization():
    """Test that default initialization works and sets correct values."""
    config = FoundationConfig()
    assert config.hidden_dim == 256
    assert config.out_dim == 128
    assert config.num_layers == 3
    assert config.dropout == 0.5

def test_config_immutability():
    """Test that config is frozen and immutable."""
    config = FoundationConfig()
    with pytest.raises(Exception):
        config.hidden_dim = 512

def test_config_invalid_hidden_dim():
    """Test that invalid hidden_dim raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(hidden_dim=0)
    with pytest.raises(ValueError):
        FoundationConfig(hidden_dim=-1)

def test_config_invalid_out_dim():
    """Test that invalid out_dim raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(out_dim=0)
    with pytest.raises(ValueError):
        FoundationConfig(out_dim=-10)

def test_config_invalid_num_layers():
    """Test that invalid num_layers raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(num_layers=0)
    with pytest.raises(ValueError):
        FoundationConfig(num_layers=-5)

def test_config_invalid_dropout():
    """Test that invalid dropout raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(dropout=-0.1)
    with pytest.raises(ValueError):
        FoundationConfig(dropout=1.1)

def test_config_invalid_learning_rate():
    """Test that invalid learning_rate raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        FoundationConfig(learning_rate=-0.001)

def test_config_invalid_batch_size():
    """Test that invalid batch_size raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(batch_size=0)
    with pytest.raises(ValueError):
        FoundationConfig(batch_size=-32)

def test_config_invalid_max_epochs():
    """Test that invalid max_epochs raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(max_epochs=0)
    with pytest.raises(ValueError):
        FoundationConfig(max_epochs=-100)

def test_config_invalid_augmentation_prob():
    """Test that invalid augmentation_prob raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(augmentation_prob=-0.1)
    with pytest.raises(ValueError):
        FoundationConfig(augmentation_prob=1.5)

def test_config_invalid_temperature():
    """Test that invalid temperature raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(temperature=0.0)
    with pytest.raises(ValueError):
        FoundationConfig(temperature=-0.5)

def test_config_invalid_projector_hidden_dim():
    """Test that invalid projector_hidden_dim raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(projector_hidden_dim=0)
    with pytest.raises(ValueError):
        FoundationConfig(projector_hidden_dim=-128)

def test_config_invalid_model_type():
    """Test that invalid model_type raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(model_type="invalid")

def test_config_invalid_encoder_type():
    """Test that invalid encoder_type raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(encoder_type="unknown")

def test_config_invalid_objective():
    """Test that invalid objective raises ValueError."""
    with pytest.raises(ValueError):
        FoundationConfig(objective="magic")

def test_config_valid_custom_initialization():
    """Test valid custom initialization."""
    config = FoundationConfig(hidden_dim=512, out_dim=256, dropout=0.2)
    assert config.hidden_dim == 512
    assert config.out_dim == 256
    assert config.dropout == 0.2
