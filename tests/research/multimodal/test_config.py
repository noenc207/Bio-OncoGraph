import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ==============================================================================
# MOCKS
# ==============================================================================
@dataclass(frozen=True)
class MultiModalConfig:
    """Configuration for multi-modal learning pipelines."""
    learning_rate: float = 0.001
    fusion_strategy: str = "early"
    batch_size: int = 32
    modality_weights: Dict[str, float] = field(default_factory=lambda: {"clinical": 1.0})
    dropout_rate: float = 0.1
    max_epochs: int = 100

    def __post_init__(self) -> None:
        if self.fusion_strategy not in ("early", "intermediate", "late", "hierarchical", "attention"):
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("Dropout rate must be between 0 and 1.")

# ==============================================================================
# TESTS
# ==============================================================================

def test_multimodal_config_default_creation() -> None:
    """Test default creation of MultiModalConfig."""
    config = MultiModalConfig()
    assert config.learning_rate == 0.001
    assert config.fusion_strategy == "early"
    assert config.batch_size == 32
    assert config.modality_weights == {"clinical": 1.0}
    assert config.dropout_rate == 0.1
    assert config.max_epochs == 100

@pytest.mark.parametrize("fusion_strategy", [
    "early", "intermediate", "late", "hierarchical", "attention"
])
def test_multimodal_config_valid_fusion_strategies(fusion_strategy: str) -> None:
    """Test creating configuration with all valid fusion strategies."""
    config = MultiModalConfig(fusion_strategy=fusion_strategy)
    assert config.fusion_strategy == fusion_strategy

@pytest.mark.parametrize("invalid_strategy", [
    "super_late", "magic", "", "EARLY", "None"
])
def test_multimodal_config_invalid_fusion_strategies(invalid_strategy: str) -> None:
    """Test that invalid fusion strategies raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported fusion strategy"):
        MultiModalConfig(fusion_strategy=invalid_strategy)

@pytest.mark.parametrize("lr", [1.0, 0.1, 0.0001, 1e-6, 50.0])
def test_multimodal_config_valid_learning_rates(lr: float) -> None:
    """Test creating config with various valid learning rates."""
    config = MultiModalConfig(learning_rate=lr)
    assert config.learning_rate == lr

@pytest.mark.parametrize("invalid_lr", [0.0, -0.1, -1.0, -1e-5])
def test_multimodal_config_invalid_learning_rates(invalid_lr: float) -> None:
    """Test that non-positive learning rates raise ValueError."""
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        MultiModalConfig(learning_rate=invalid_lr)

@pytest.mark.parametrize("batch_size", [1, 2, 32, 64, 1024])
def test_multimodal_config_valid_batch_sizes(batch_size: int) -> None:
    """Test valid batch size initialization."""
    config = MultiModalConfig(batch_size=batch_size)
    assert config.batch_size == batch_size

@pytest.mark.parametrize("invalid_batch", [0, -1, -32])
def test_multimodal_config_invalid_batch_sizes(invalid_batch: int) -> None:
    """Test invalid batch sizes raise ValueError."""
    with pytest.raises(ValueError, match="Batch size must be positive"):
        MultiModalConfig(batch_size=invalid_batch)

@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_multimodal_config_valid_dropout_rates(dropout: float) -> None:
    """Test valid dropout rates."""
    config = MultiModalConfig(dropout_rate=dropout)
    assert config.dropout_rate == dropout

@pytest.mark.parametrize("invalid_dropout", [-0.1, 1.1, 10.0, -5.0])
def test_multimodal_config_invalid_dropout_rates(invalid_dropout: float) -> None:
    """Test invalid dropout rates raise ValueError."""
    with pytest.raises(ValueError, match="Dropout rate must be between 0 and 1"):
        MultiModalConfig(dropout_rate=invalid_dropout)

def test_multimodal_config_immutability() -> None:
    """Test that the dataclass is frozen and cannot be mutated."""
    config = MultiModalConfig()
    with pytest.raises(Exception): # FrozenInstanceError
        config.learning_rate = 0.01 # type: ignore

@pytest.mark.parametrize("weights", [
    {"clinical": 1.0, "genomics": 0.5},
    {"histopathology": 2.0},
    {}
])
def test_multimodal_config_modality_weights(weights: Dict[str, float]) -> None:
    """Test initialization of modality weights."""
    config = MultiModalConfig(modality_weights=weights)
    assert config.modality_weights == weights
