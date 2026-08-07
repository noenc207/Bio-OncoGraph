import pytest
import dataclasses
from typing import Any, Dict

from bio_oncograph.research.histograph.config import HistographConfig


def test_config_initialization_defaults() -> None:
    """Test that HistographConfig initializes with expected default values."""
    config = HistographConfig()
    assert config is not None


def test_config_default_knn_k() -> None:
    """Test default k for KNN."""
    config = HistographConfig()
    assert config.knn_k == 5


def test_config_default_radius() -> None:
    """Test default radius."""
    config = HistographConfig()
    assert config.radius > 0.0


def test_config_custom_initialization() -> None:
    """Test initialization with custom parameters."""
    config = HistographConfig(knn_k=10, radius=50.0, edge_strategy="radius")
    assert config.knn_k == 10
    assert config.radius == 50.0
    assert config.edge_strategy == "radius"


def test_config_is_frozen() -> None:
    """Test that HistographConfig is immutable (frozen)."""
    config = HistographConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.knn_k = 10


def test_config_invalid_knn_k_zero() -> None:
    """Test that setting knn_k to 0 raises ValueError."""
    with pytest.raises(ValueError):
        HistographConfig(knn_k=0)


def test_config_invalid_knn_k_negative() -> None:
    """Test that setting knn_k to a negative number raises ValueError."""
    with pytest.raises(ValueError):
        HistographConfig(knn_k=-5)


def test_config_invalid_radius_zero() -> None:
    """Test that setting radius to 0 raises ValueError."""
    with pytest.raises(ValueError):
        HistographConfig(radius=0.0)


def test_config_invalid_radius_negative() -> None:
    """Test that setting radius to a negative number raises ValueError."""
    with pytest.raises(ValueError):
        HistographConfig(radius=-10.0)


def test_config_invalid_edge_strategy() -> None:
    """Test that an unknown edge strategy raises ValueError."""
    with pytest.raises(ValueError):
        HistographConfig(edge_strategy="unknown_magic_strategy")


def test_config_to_dict() -> None:
    """Test converting config to dictionary."""
    config = HistographConfig(knn_k=7)
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["knn_k"] == 7


def test_config_from_dict() -> None:
    """Test creating config from dictionary."""
    d = {"knn_k": 8, "radius": 25.0}
    config = HistographConfig.from_dict(d)
    assert config.knn_k == 8
    assert config.radius == 25.0


def test_config_equality() -> None:
    """Test equality operator for identical configs."""
    config1 = HistographConfig(knn_k=5)
    config2 = HistographConfig(knn_k=5)
    assert config1 == config2


def test_config_inequality() -> None:
    """Test inequality operator for different configs."""
    config1 = HistographConfig(knn_k=5)
    config2 = HistographConfig(knn_k=10)
    assert config1 != config2


def test_config_hashable() -> None:
    """Test that config can be used as a dictionary key (is hashable)."""
    config = HistographConfig()
    d = {config: "value"}
    assert d[config] == "value"
