"""Tests for explainability engine."""
import pytest
import numpy as np
from bio_oncograph.ai.explainability.config import ExplainabilityConfig
from bio_oncograph.ai.explainability.explainer import ExplainabilityEngine
from bio_oncograph.ai.explainability.outputs import ExplainabilityOutput, UncertaintyOutput, TrustScoreOutput
from bio_oncograph.ai.explainability.report import ExplainabilityReport

class MockModel:
    def predict(self, x):
        return np.ones(len(x)) * 0.8

@pytest.fixture
def engine():
    config = ExplainabilityConfig()
    model = MockModel()
    return ExplainabilityEngine(model=model, config=config)

@pytest.fixture
def sample_inputs():
    return {"features": np.array([0.5, 0.2]), "model_output": np.array([0.8])}

def test_engine_init():
    """Test ExplainabilityEngine initialization."""
    config = ExplainabilityConfig()
    model = MockModel()
    engine = ExplainabilityEngine(model=model, config=config)
    assert engine.model is not None
    assert engine.config is not None

def test_engine_explain(engine, sample_inputs):
    """Test engine explain method."""
    output = engine.explain(sample_inputs)
    assert isinstance(output, ExplainabilityOutput)
    assert output.feature_importances is not None

def test_engine_estimate_uncertainty(engine, sample_inputs):
    """Test engine estimate_uncertainty method."""
    output = engine.estimate_uncertainty(sample_inputs)
    assert isinstance(output, UncertaintyOutput)
    assert output.total >= 0

def test_engine_compute_trust_score(engine, sample_inputs):
    """Test engine compute_trust_score method."""
    output = engine.compute_trust_score(sample_inputs)
    assert isinstance(output, TrustScoreOutput)
    assert 0 <= output.score <= 1

def test_engine_generate_report(engine, sample_inputs):
    """Test engine full report flow."""
    report = engine.generate_report(sample_inputs)
    assert isinstance(report, ExplainabilityReport)
    assert isinstance(report.explanation, ExplainabilityOutput)
    assert isinstance(report.uncertainty, UncertaintyOutput)
    assert isinstance(report.trust_score, TrustScoreOutput)

def test_engine_invalid_inputs(engine):
    """Test engine with invalid inputs."""
    bad_inputs = {"features": np.array([np.nan]), "model_output": np.array([0.8])}
    with pytest.raises((ValueError, TypeError)):
        engine.explain(bad_inputs)

def test_engine_missing_inputs(engine):
    """Test engine with missing inputs."""
    with pytest.raises((ValueError, TypeError, KeyError)):
        engine.explain({})

def test_engine_explain_batch(engine):
    """Test engine explain with batch data."""
    inputs = {"features": np.array([[0.5, 0.2], [0.1, 0.9]]), "model_output": np.array([0.8, 0.2])}
    output = engine.explain(inputs)
    assert isinstance(output, ExplainabilityOutput)
    assert len(output.feature_importances) > 0

def test_engine_config_override():
    """Test engine initialization with config override."""
    config = ExplainabilityConfig(num_samples=100)
    engine = ExplainabilityEngine(model=MockModel(), config=config)
    assert engine.config.num_samples == 100
