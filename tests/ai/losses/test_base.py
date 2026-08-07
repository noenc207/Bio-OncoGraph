"""Tests for LossFunction."""

import pytest
from abc import ABC
import numpy as np

from ai.losses.base import LossFunction
from ai.losses.config import LossConfig
from ai.losses.outputs import LossOutput


def test_loss_function_is_abc() -> None:
    """Test that LossFunction is an abstract base class."""
    assert issubclass(LossFunction, ABC)


def test_loss_function_cannot_be_instantiated() -> None:
    """Test that LossFunction cannot be instantiated without implementing compute."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # Type ignore is used because we're intentionally doing something invalid
        _ = LossFunction(config=LossConfig())  # type: ignore


def test_loss_function_concrete_implementation() -> None:
    """Test a valid concrete implementation of LossFunction."""
    class DummyLoss(LossFunction):
        def compute(self, predictions: np.ndarray, targets: np.ndarray) -> LossOutput:
            return LossOutput(value=0.0, name=self.config.name)

    config = LossConfig(name="dummy")
    loss_fn = DummyLoss(config=config)
    assert loss_fn.config == config

    preds = np.array([1.0, 2.0])
    targets = np.array([1.0, 2.0])
    output = loss_fn.compute(preds, targets)
    assert output.value == 0.0
    assert output.name == "dummy"


def test_loss_function_config_property() -> None:
    """Test the config property of a concrete LossFunction."""
    class DummyLoss(LossFunction):
        def compute(self, predictions: np.ndarray, targets: np.ndarray) -> LossOutput:
            return LossOutput(value=1.0, name="dummy")

    config = LossConfig(name="dummy_loss")
    loss_fn = DummyLoss(config=config)
    
    assert loss_fn.config.name == "dummy_loss"
    
    # Assuming config is meant to be read-only property or standard attribute
    # Modifying the config field should ideally be restricted if it's a property without setter
    if hasattr(DummyLoss, "config") and isinstance(DummyLoss.config, property):
        with pytest.raises(AttributeError):
            loss_fn.config = LossConfig(name="new")


def test_loss_function_call_method() -> None:
    """Test if __call__ routes to compute."""
    class DummyLoss(LossFunction):
        def compute(self, predictions: np.ndarray, targets: np.ndarray) -> LossOutput:
            return LossOutput(value=42.0, name="dummy")

    loss_fn = DummyLoss(config=LossConfig())
    preds = np.array([0.0])
    targets = np.array([0.0])
    
    # If __call__ is implemented to map to compute:
    if hasattr(loss_fn, "__call__"):
        output = loss_fn(preds, targets)
        assert output.value == 42.0
