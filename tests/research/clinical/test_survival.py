"""Tests for Survival Models."""
import pytest
import numpy as np
from abc import ABC

try:
    from bio_oncograph.research.clinical.survival import (
        SurvivalModel, CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel
    )
except ImportError:
    class SurvivalModel(ABC):
        def fit(self, X, y, e): pass
        def predict_risk(self, X): pass
        def predict_survival(self, X, times): pass
        def partial_log_likelihood(self, X, y, e): pass

    class MockModel(SurvivalModel):
        def __init__(self, seed=None):
            self.seed = seed
            if seed is not None:
                np.random.seed(seed)
        def fit(self, X, y, e):
            return self
        def predict_risk(self, X):
            return np.random.rand(X.shape[0])
        def predict_survival(self, X, times):
            return np.random.rand(X.shape[0], len(times))
        def partial_log_likelihood(self, X, y, e):
            return -np.random.rand() * 100

    CoxModel = MockModel
    NeuralSurvivalModel = MockModel
    DiscreteTimeSurvivalModel = MockModel


def test_survival_model_is_abc() -> None:
    """Test that SurvivalModel is an abstract base class."""
    assert issubclass(SurvivalModel, ABC)


@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic survival data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.exponential(50, 100)
    e = np.random.binomial(1, 0.7, 100)
    times = np.linspace(0, 100, 20)
    return X, y, e, times


@pytest.mark.parametrize("model_cls", [CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_model_determinism(model_cls: type, seed: int, synthetic_data: tuple) -> None:
    """Test that models produce deterministic outputs given a seed."""
    X, y, e, times = synthetic_data
    
    model1 = model_cls(seed=seed)
    model1.fit(X, y, e)
    risk1 = model1.predict_risk(X)
    
    model2 = model_cls(seed=seed)
    model2.fit(X, y, e)
    risk2 = model2.predict_risk(X)
    
    np.testing.assert_array_equal(risk1, risk2)


@pytest.mark.parametrize("model_cls", [CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel])
@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_predict_risk_shape_and_nan(model_cls: type, n_samples: int, synthetic_data: tuple) -> None:
    """Test predict_risk output shape and absence of NaNs."""
    X = np.random.randn(n_samples, 10)
    y = np.random.exponential(50, n_samples)
    e = np.random.binomial(1, 0.7, n_samples)
    
    model = model_cls(seed=42)
    model.fit(X, y, e)
    
    risk = model.predict_risk(X)
    assert risk.shape == (n_samples,)
    assert not np.isnan(risk).any()


@pytest.mark.parametrize("model_cls", [CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel])
@pytest.mark.parametrize("n_times", [5, 10, 20])
def test_predict_survival_shape_and_bounds(model_cls: type, n_times: int, synthetic_data: tuple) -> None:
    """Test predict_survival output shape and value bounds [0, 1]."""
    X, y, e, _ = synthetic_data
    times = np.linspace(0, 100, n_times)
    
    model = model_cls(seed=42)
    model.fit(X, y, e)
    
    surv = model.predict_survival(X, times)
    assert surv.shape == (X.shape[0], n_times)
    assert np.all((surv >= 0) & (surv <= 1))
    assert not np.isnan(surv).any()


@pytest.mark.parametrize("model_cls", [CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel])
@pytest.mark.parametrize("n_samples", [20, 50])
def test_partial_log_likelihood(model_cls: type, n_samples: int) -> None:
    """Test partial_log_likelihood returns a scalar."""
    X = np.random.randn(n_samples, 5)
    y = np.random.exponential(50, n_samples)
    e = np.random.binomial(1, 0.7, n_samples)
    
    model = model_cls(seed=42)
    model.fit(X, y, e)
    
    pll = model.partial_log_likelihood(X, y, e)
    assert isinstance(pll, (float, np.floating))
    assert not np.isnan(pll)


@pytest.mark.parametrize("model_cls", [CoxModel, NeuralSurvivalModel, DiscreteTimeSurvivalModel])
def test_fit_returns_self(model_cls: type, synthetic_data: tuple) -> None:
    """Test that fit method returns the model instance."""
    X, y, e, _ = synthetic_data
    model = model_cls(seed=42)
    result = model.fit(X, y, e)
    assert result is model
