"""Tests for explainability output classes."""
import pytest
import numpy as np
from dataclasses import FrozenInstanceError
from bio_oncograph.ai.explainability.outputs import ExplainabilityOutput, UncertaintyOutput, TrustScoreOutput

def test_explainability_output_creation():
    """Test creation of ExplainabilityOutput."""
    out = ExplainabilityOutput(feature_importances=np.array([0.1, 0.9]), method="ig")
    assert out is not None

def test_explainability_output_immutability():
    """Test ExplainabilityOutput immutability."""
    out = ExplainabilityOutput(feature_importances=np.array([0.1]), method="ig")
    with pytest.raises(FrozenInstanceError):
        out.method = "lime"

def test_explainability_output_values():
    """Test ExplainabilityOutput values."""
    arr = np.array([0.5, 0.5])
    out = ExplainabilityOutput(feature_importances=arr, method="ig")
    np.testing.assert_array_equal(out.feature_importances, arr)
    assert out.method == "ig"

def test_uncertainty_output_creation():
    """Test creation of UncertaintyOutput."""
    out = UncertaintyOutput(epistemic=0.1, aleatoric=0.2, total=0.3)
    assert out is not None

def test_uncertainty_output_immutability():
    """Test UncertaintyOutput immutability."""
    out = UncertaintyOutput(epistemic=0.1, aleatoric=0.2, total=0.3)
    with pytest.raises(FrozenInstanceError):
        out.total = 0.4

def test_uncertainty_output_values():
    """Test UncertaintyOutput values."""
    out = UncertaintyOutput(epistemic=0.1, aleatoric=0.2, total=0.3)
    assert out.epistemic == 0.1
    assert out.aleatoric == 0.2
    assert out.total == 0.3

def test_trust_score_output_creation():
    """Test creation of TrustScoreOutput."""
    out = TrustScoreOutput(score=0.95, is_trustworthy=True)
    assert out is not None

def test_trust_score_output_immutability():
    """Test TrustScoreOutput immutability."""
    out = TrustScoreOutput(score=0.95, is_trustworthy=True)
    with pytest.raises(FrozenInstanceError):
        out.score = 0.5

def test_trust_score_output_values():
    """Test TrustScoreOutput values."""
    out = TrustScoreOutput(score=0.95, is_trustworthy=True)
    assert out.score == 0.95
    assert out.is_trustworthy is True
