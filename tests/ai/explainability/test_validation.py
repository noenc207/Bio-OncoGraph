"""Tests for explainability validation."""
import pytest
import numpy as np
from bio_oncograph.ai.explainability.validation import validate_explainability_inputs

def test_validate_valid_inputs():
    """Test validation with valid inputs."""
    inputs = {"features": np.array([1.0, 2.0]), "model_output": np.array([0.8])}
    assert validate_explainability_inputs(inputs) is True

def test_validate_missing_keys():
    """Test validation with missing keys."""
    inputs = {"features": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="missing|required"):
        validate_explainability_inputs(inputs)

def test_validate_nan_features():
    """Test validation catches NaN in features."""
    inputs = {"features": np.array([1.0, np.nan]), "model_output": np.array([0.8])}
    with pytest.raises(ValueError, match="NaN|nan"):
        validate_explainability_inputs(inputs)

def test_validate_inf_features():
    """Test validation catches Inf in features."""
    inputs = {"features": np.array([1.0, np.inf]), "model_output": np.array([0.8])}
    with pytest.raises(ValueError, match="Inf|infinity|inf"):
        validate_explainability_inputs(inputs)

def test_validate_nan_output():
    """Test validation catches NaN in output."""
    inputs = {"features": np.array([1.0, 2.0]), "model_output": np.array([np.nan])}
    with pytest.raises(ValueError, match="NaN|nan"):
        validate_explainability_inputs(inputs)

def test_validate_inf_output():
    """Test validation catches Inf in output."""
    inputs = {"features": np.array([1.0, 2.0]), "model_output": np.array([np.inf])}
    with pytest.raises(ValueError, match="Inf|infinity|inf"):
        validate_explainability_inputs(inputs)

def test_validate_empty_dict():
    """Test validation with empty dictionary."""
    with pytest.raises(ValueError, match="missing|empty"):
        validate_explainability_inputs({})

def test_validate_invalid_type():
    """Test validation with non-dict input."""
    with pytest.raises((TypeError, ValueError)):
        validate_explainability_inputs(list())
