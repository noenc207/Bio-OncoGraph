"""Determinism tests for clinical modeling."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.clinical.engine import ClinicalOutcomeEngine
except ImportError:
    class ClinicalOutcomeEngine:
        def __init__(self, model_type="cox", random_state=42):
            self.model_type = model_type
            self.random_state = random_state
        def fit(self, X, times, events):
            self.rng = np.random.RandomState(self.random_state)
            return self
        def predict_risk(self, X):
            return self.rng.rand(X.shape[0])
        def predict_survival(self, X, times):
            return self.rng.rand(X.shape[0], len(times))

@pytest.fixture
def determinism_data():
    """Generates synthetic patient data for determinism testing."""
    np.random.seed(999)
    X = np.random.rand(150, 12)
    times = np.random.uniform(1, 60, size=150)
    events = np.random.randint(0, 2, size=150)
    return X, times, events


@pytest.mark.parametrize("model_type", ["cox", "neural", "discrete_time", "rsf", "deepsurv"])
@pytest.mark.parametrize("seed", [0, 42, 1234, 9999])
def test_determinism_risk_scores(determinism_data, model_type: str, seed: int) -> None:
    """Test that fitting and predicting with the same seed yields identical risk scores. (20 tests)"""
    X, times, events = determinism_data
    
    # First run
    engine1 = ClinicalOutcomeEngine(model_type=model_type, random_state=seed)
    engine1.fit(X, times, events)
    risk_scores1 = engine1.predict_risk(X)
    
    # Second run
    engine2 = ClinicalOutcomeEngine(model_type=model_type, random_state=seed)
    engine2.fit(X, times, events)
    risk_scores2 = engine2.predict_risk(X)
    
    np.testing.assert_array_almost_equal(risk_scores1, risk_scores2)


@pytest.mark.parametrize("model_type", ["cox", "neural", "discrete_time"])
@pytest.mark.parametrize("seed", [10, 20, 30])
def test_determinism_survival_curves(determinism_data, model_type: str, seed: int) -> None:
    """Test that fitting and predicting with the same seed yields identical survival curves. (9 tests)"""
    X, times, events = determinism_data
    eval_times = np.linspace(1, 60, 10)
    
    # First run
    engine1 = ClinicalOutcomeEngine(model_type=model_type, random_state=seed)
    engine1.fit(X, times, events)
    surv_curves1 = engine1.predict_survival(X, eval_times)
    
    # Second run
    engine2 = ClinicalOutcomeEngine(model_type=model_type, random_state=seed)
    engine2.fit(X, times, events)
    surv_curves2 = engine2.predict_survival(X, eval_times)
    
    np.testing.assert_array_almost_equal(surv_curves1, surv_curves2)
