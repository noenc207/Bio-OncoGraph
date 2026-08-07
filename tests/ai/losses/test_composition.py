"""Tests for CompositeLoss."""

import pytest
import numpy as np

from ai.losses.composition import CompositeLoss
from ai.losses.base import LossFunction
from ai.losses.outputs import LossOutput
from ai.losses.config import LossConfig


class DummyLoss(LossFunction):
    """Dummy loss function for testing."""
    def __init__(self, name: str, fixed_value: float):
        super().__init__(config=LossConfig(name=name))
        self.fixed_value = fixed_value

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> LossOutput:
        return LossOutput(value=self.fixed_value, name=self.config.name)


def test_composite_loss_creation() -> None:
    """Test creation of CompositeLoss with multiple components."""
    loss1 = DummyLoss("loss1", 1.0)
    loss2 = DummyLoss("loss2", 2.0)
    
    comp_loss = CompositeLoss(losses={"l1": loss1, "l2": loss2}, weights={"l1": 1.0, "l2": 0.5})
    assert len(comp_loss.losses) == 2
    assert comp_loss.weights["l2"] == 0.5


def test_composite_loss_weighted_sum() -> None:
    """Test that the weighted sum is computed correctly."""
    loss1 = DummyLoss("loss1", 2.0)
    loss2 = DummyLoss("loss2", 4.0)
    
    comp_loss = CompositeLoss(losses={"l1": loss1, "l2": loss2}, weights={"l1": 0.5, "l2": 0.25})
    
    preds = np.array([0.0])
    targets = np.array([0.0])
    
    output = comp_loss.compute(preds, targets)
    # Expected: (2.0 * 0.5) + (4.0 * 0.25) = 1.0 + 1.0 = 2.0
    assert output.value == 2.0


def test_composite_loss_components_reported() -> None:
    """Test that component losses are reported in the output metadata."""
    loss1 = DummyLoss("loss1", 2.0)
    
    comp_loss = CompositeLoss(losses={"l1": loss1}, weights={"l1": 0.5})
    preds = np.array([0.0])
    targets = np.array([0.0])
    
    output = comp_loss.compute(preds, targets)
    
    assert "components" in output.metadata
    components = output.metadata["components"]
    assert components["l1"] == 2.0  # Usually records unweighted or weighted, assume unweighted or specific format
    # Let's assert it exists in the dict
    assert "l1" in components


def test_composite_loss_missing_weights() -> None:
    """Test that missing weights raise an error."""
    loss1 = DummyLoss("loss1", 1.0)
    
    with pytest.raises((ValueError, KeyError)):
        # Provide losses but mismatching or missing weights keys
        CompositeLoss(losses={"l1": loss1}, weights={"l2": 1.0})


def test_composite_loss_missing_losses() -> None:
    """Test that missing losses raise an error."""
    loss1 = DummyLoss("loss1", 1.0)
    
    with pytest.raises((ValueError, KeyError)):
        # Weights provided for non-existent loss
        CompositeLoss(losses={"l1": loss1}, weights={"l1": 1.0, "l2": 1.0})


def test_composite_loss_disable_via_zero_weight() -> None:
    """Test that a component with weight 0 is effectively disabled."""
    loss1 = DummyLoss("loss1", 10.0)
    loss2 = DummyLoss("loss2", 5.0)
    
    comp_loss = CompositeLoss(losses={"l1": loss1, "l2": loss2}, weights={"l1": 1.0, "l2": 0.0})
    
    preds = np.array([0.0])
    targets = np.array([0.0])
    
    output = comp_loss.compute(preds, targets)
    # Expected: 10.0 * 1.0 + 5.0 * 0.0 = 10.0
    assert output.value == 10.0
    
    # Ideally, disabled losses might not be computed or just result in 0
    # Both behaviors satisfy "disabled" via weight 0
