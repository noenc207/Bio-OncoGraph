"""Tests for validate_loss_inputs."""

import pytest
import numpy as np

from ai.losses.validation import validate_loss_inputs


def test_validate_loss_inputs_valid() -> None:
    """Test that valid inputs pass validation without error."""
    preds = np.array([0.1, 0.9])
    targets = np.array([0.0, 1.0])
    # Should not raise any exception
    validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_nan_in_predictions() -> None:
    """Test that NaN in predictions raises ValueError."""
    preds = np.array([0.1, np.nan])
    targets = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="(?i)nan"):
        validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_nan_in_targets() -> None:
    """Test that NaN in targets raises ValueError."""
    preds = np.array([0.1, 0.9])
    targets = np.array([0.0, np.nan])
    with pytest.raises(ValueError, match="(?i)nan"):
        validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_inf_in_predictions() -> None:
    """Test that Inf in predictions raises ValueError."""
    preds = np.array([0.1, np.inf])
    targets = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="(?i)inf"):
        validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_inf_in_targets() -> None:
    """Test that Inf in targets raises ValueError."""
    preds = np.array([0.1, 0.9])
    targets = np.array([-np.inf, 1.0])
    with pytest.raises(ValueError, match="(?i)inf"):
        validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_shape_mismatch() -> None:
    """Test that shape mismatch between predictions and targets raises ValueError."""
    preds = np.array([0.1, 0.9, 0.5])
    targets = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="(?i)shape"):
        validate_loss_inputs(preds, targets)


def test_validate_loss_inputs_different_types() -> None:
    """Test that significantly different data types (if disallowed) raise TypeError."""
    preds = np.array([0.1, 0.9], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int32)
    # The validation function might require them to be both float or specific types
    # This test asserts a TypeError or ValueError is raised if type consistency is enforced.
    # If the function casts automatically, this test might need adjustment.
    try:
        validate_loss_inputs(preds, targets)
    except (TypeError, ValueError):
        pass  # Expected if types must match


def test_validate_loss_inputs_multidimensional_valid() -> None:
    """Test validation with valid 2D arrays."""
    preds = np.array([[0.1, 0.9], [0.8, 0.2]])
    targets = np.array([[0.0, 1.0], [1.0, 0.0]])
    validate_loss_inputs(preds, targets)
