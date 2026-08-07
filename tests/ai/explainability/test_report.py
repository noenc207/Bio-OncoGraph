"""Tests for explainability report."""
import pytest
import numpy as np
from dataclasses import FrozenInstanceError
from bio_oncograph.ai.explainability.outputs import ExplainabilityOutput, UncertaintyOutput, TrustScoreOutput
from bio_oncograph.ai.explainability.report import ExplainabilityReport

@pytest.fixture
def sample_report_data():
    return {
        "explanation": ExplainabilityOutput(feature_importances=np.array([1.0]), method="ig"),
        "uncertainty": UncertaintyOutput(epistemic=0.1, aleatoric=0.1, total=0.2),
        "trust_score": TrustScoreOutput(score=0.9, is_trustworthy=True),
        "metadata": {"version": "1.0"}
    }

def test_report_creation(sample_report_data):
    """Test ExplainabilityReport creation."""
    report = ExplainabilityReport(**sample_report_data)
    assert report is not None

def test_report_immutability(sample_report_data):
    """Test ExplainabilityReport immutability."""
    report = ExplainabilityReport(**sample_report_data)
    with pytest.raises(FrozenInstanceError):
        report.metadata = {}

def test_report_values(sample_report_data):
    """Test ExplainabilityReport values."""
    report = ExplainabilityReport(**sample_report_data)
    assert report.explanation.method == "ig"
    assert report.uncertainty.total == 0.2
    assert report.trust_score.score == 0.9
    assert report.metadata["version"] == "1.0"

def test_report_missing_metadata(sample_report_data):
    """Test report with missing optional metadata."""
    del sample_report_data["metadata"]
    report = ExplainabilityReport(**sample_report_data, metadata=None)
    assert report.metadata is None

def test_report_repr(sample_report_data):
    """Test report representation."""
    report = ExplainabilityReport(**sample_report_data)
    assert "ExplainabilityReport" in repr(report)

def test_report_explanation_type(sample_report_data):
    """Test report explanation type."""
    report = ExplainabilityReport(**sample_report_data)
    assert isinstance(report.explanation, ExplainabilityOutput)

def test_report_uncertainty_type(sample_report_data):
    """Test report uncertainty type."""
    report = ExplainabilityReport(**sample_report_data)
    assert isinstance(report.uncertainty, UncertaintyOutput)

def test_report_trust_score_type(sample_report_data):
    """Test report trust score type."""
    report = ExplainabilityReport(**sample_report_data)
    assert isinstance(report.trust_score, TrustScoreOutput)
