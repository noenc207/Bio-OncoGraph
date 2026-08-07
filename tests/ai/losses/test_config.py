"""Tests for LossConfig."""

import pytest
from dataclasses import FrozenInstanceError
from typing import Any

# Adjust import according to actual project structure if needed.
# Assuming ai.losses.config.LossConfig exists and is a frozen dataclass.
from ai.losses.config import LossConfig


def test_loss_config_creation() -> None:
    """Test standard creation of LossConfig."""
    config = LossConfig(name="mse", weight=0.5, reduction="sum")
    assert config.name == "mse"
    assert config.weight == 0.5
    assert config.reduction == "sum"


def test_loss_config_defaults() -> None:
    """Test default values of LossConfig."""
    config = LossConfig()
    assert config.weight == 1.0
    assert config.reduction == "mean"
    # Default name might vary, assuming 'loss' or similar
    assert isinstance(config.name, str)


def test_loss_config_immutability_name() -> None:
    """Test immutability of the name field."""
    config = LossConfig(name="test")
    with pytest.raises(FrozenInstanceError):
        config.name = "new_name"


def test_loss_config_immutability_weight() -> None:
    """Test immutability of the weight field."""
    config = LossConfig(weight=1.0)
    with pytest.raises(FrozenInstanceError):
        config.weight = 2.0


def test_loss_config_immutability_reduction() -> None:
    """Test immutability of the reduction field."""
    config = LossConfig(reduction="mean")
    with pytest.raises(FrozenInstanceError):
        config.reduction = "sum"


def test_loss_config_fields() -> None:
    """Test that all expected fields are present and correctly typed."""
    config = LossConfig(name="l1", weight=2.0, reduction="none")
    assert hasattr(config, "name")
    assert hasattr(config, "weight")
    assert hasattr(config, "reduction")
    assert isinstance(config.name, str)
    assert isinstance(config.weight, float)
    assert isinstance(config.reduction, str)

def test_loss_config_hash() -> None:
    """Test that LossConfig is hashable."""
    config1 = LossConfig(name="mse")
    config2 = LossConfig(name="mse")
    assert hash(config1) == hash(config2)

def test_loss_config_equality() -> None:
    """Test equality between config instances."""
    config1 = LossConfig(name="mse", weight=1.0)
    config2 = LossConfig(name="mse", weight=1.0)
    config3 = LossConfig(name="mae", weight=1.0)
    assert config1 == config2
    assert config1 != config3
