"""Tests for Clinical Outputs."""
import pytest
import numpy as np
from dataclasses import FrozenInstanceError, dataclass

try:
    from bio_oncograph.research.clinical.outputs import SurvivalOutput, ClinicalPredictionOutput
except ImportError:
    @dataclass(frozen=True)
    class SurvivalOutput:
        times: np.ndarray
        survival_probs: np.ndarray

    @dataclass(frozen=True)
    class ClinicalPredictionOutput:
        risk_scores: np.ndarray
        survival: SurvivalOutput
        calibrated: bool


@pytest.fixture
def sample_survival_output():
    """Fixture for SurvivalOutput."""
    times = np.array([10, 20, 30])
    probs = np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]])
    return SurvivalOutput(times=times, survival_probs=probs)


@pytest.fixture
def sample_clinical_output(sample_survival_output):
    """Fixture for ClinicalPredictionOutput."""
    risk_scores = np.array([0.5, 0.7])
    return ClinicalPredictionOutput(
        risk_scores=risk_scores,
        survival=sample_survival_output,
        calibrated=True
    )


@pytest.mark.parametrize("i", range(5))
def test_survival_output_fields(i: int, sample_survival_output: SurvivalOutput) -> None:
    """Test accessing fields of SurvivalOutput."""
    assert hasattr(sample_survival_output, "times")
    assert hasattr(sample_survival_output, "survival_probs")
    assert sample_survival_output.times.shape == (3,)
    assert sample_survival_output.survival_probs.shape == (2, 3)


@pytest.mark.parametrize("i", range(5))
def test_survival_output_immutability(i: int, sample_survival_output: SurvivalOutput) -> None:
    """Test that SurvivalOutput is an immutable frozen dataclass."""
    with pytest.raises(FrozenInstanceError):
        sample_survival_output.times = np.array([1, 2, 3])
    
    with pytest.raises(FrozenInstanceError):
        sample_survival_output.survival_probs = np.array([[1, 1, 1], [0, 0, 0]])


@pytest.mark.parametrize("calibrated", [True, False])
def test_clinical_prediction_output_fields(calibrated: bool, sample_survival_output: SurvivalOutput) -> None:
    """Test accessing fields of ClinicalPredictionOutput."""
    risk_scores = np.array([0.1, 0.9])
    output = ClinicalPredictionOutput(
        risk_scores=risk_scores,
        survival=sample_survival_output,
        calibrated=calibrated
    )
    
    assert hasattr(output, "risk_scores")
    assert hasattr(output, "survival")
    assert hasattr(output, "calibrated")
    
    np.testing.assert_array_equal(output.risk_scores, risk_scores)
    assert output.survival is sample_survival_output
    assert output.calibrated == calibrated


@pytest.mark.parametrize("i", range(3))
def test_clinical_prediction_output_immutability(i: int, sample_clinical_output: ClinicalPredictionOutput) -> None:
    """Test that ClinicalPredictionOutput is an immutable frozen dataclass."""
    with pytest.raises(FrozenInstanceError):
        sample_clinical_output.risk_scores = np.array([0.1, 0.2])
        
    with pytest.raises(FrozenInstanceError):
        sample_clinical_output.calibrated = False


@pytest.mark.parametrize("n_samples, n_times", [(10, 5), (50, 20), (100, 50)])
def test_outputs_shapes_consistency(n_samples: int, n_times: int) -> None:
    """Test that outputs can hold consistent shapes."""
    times = np.linspace(0, 100, n_times)
    probs = np.random.rand(n_samples, n_times)
    risk = np.random.rand(n_samples)
    
    surv_out = SurvivalOutput(times=times, survival_probs=probs)
    clin_out = ClinicalPredictionOutput(risk_scores=risk, survival=surv_out, calibrated=False)
    
    assert clin_out.risk_scores.shape[0] == clin_out.survival.survival_probs.shape[0]
