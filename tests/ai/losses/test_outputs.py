"""Tests for LossOutput."""

import pytest
from dataclasses import FrozenInstanceError
import numpy as np

from ai.losses.outputs import LossOutput


def test_loss_output_creation() -> None:
    """Test standard creation of LossOutput."""
    output = LossOutput(value=1.5, name="mse", metadata={"reduction": "mean"})
    assert output.value == 1.5
    assert output.name == "mse"
    assert output.metadata == {"reduction": "mean"}


def test_loss_output_creation_numpy_value() -> None:
    """Test creation with a numpy scalar."""
    output = LossOutput(value=np.float32(2.5), name="l1")
    assert np.isclose(output.value, 2.5)


def test_loss_output_defaults() -> None:
    """Test default values for LossOutput."""
    output = LossOutput(value=0.0, name="test")
    assert output.metadata == {}


def test_loss_output_immutability_value() -> None:
    """Test immutability of the value field."""
    output = LossOutput(value=1.0, name="test")
    with pytest.raises(FrozenInstanceError):
        output.value = 2.0


def test_loss_output_immutability_name() -> None:
    """Test immutability of the name field."""
    output = LossOutput(value=1.0, name="test")
    with pytest.raises(FrozenInstanceError):
        output.name = "new_name"


def test_loss_output_immutability_metadata() -> None:
    """Test immutability of the metadata field."""
    output = LossOutput(value=1.0, name="test", metadata={"info": "val"})
    with pytest.raises(FrozenInstanceError):
        output.metadata = {"new": "val"}


def test_loss_output_properties() -> None:
    """Test property accessors if any exist (like formatted strings)."""
    output = LossOutput(value=3.14159, name="custom")
    assert "custom" in str(output)
    assert hasattr(output, "value")
    assert hasattr(output, "name")
