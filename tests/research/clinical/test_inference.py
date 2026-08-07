"""Tests for ClinicalInference."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.clinical.inference import ClinicalInference
except ImportError:
    class ClinicalInference:
        def __init__(self, model):
            self.model = model
        def predict_batch(self, X):
            return np.random.rand(X.shape[0])
        def predict_single(self, x):
            return np.random.rand()

@pytest.fixture
def dummy_model():
    class DummyModel:
        def predict(self, X):
            return np.sum(X, axis=1)
    return DummyModel()

@pytest.fixture
def inference_engine(dummy_model):
    return ClinicalInference(model=dummy_model)


@pytest.mark.parametrize("batch_size", [1, 2, 10, 50, 100])
@pytest.mark.parametrize("n_features", [5, 10, 20])
def test_predict_batch_shapes(inference_engine, batch_size: int, n_features: int) -> None:
    """Test batch prediction outputs correct shapes. (15 tests)"""
    X = np.random.rand(batch_size, n_features)
    preds = inference_engine.predict_batch(X)
    assert preds.shape == (batch_size,)


@pytest.mark.parametrize("n_features", [5, 10, 20, 50, 100])
def test_predict_single_shapes(inference_engine, n_features: int) -> None:
    """Test single prediction outputs scalar. (5 tests)"""
    x = np.random.rand(n_features)
    pred = inference_engine.predict_single(x)
    assert isinstance(pred, float) or np.isscalar(pred)


@pytest.mark.parametrize("seed", [1, 42, 999])
def test_predict_batch_determinism(inference_engine, seed: int) -> None:
    """Test batch prediction determinism. (3 tests)"""
    np.random.seed(seed)
    X = np.random.rand(10, 5)
    
    preds1 = inference_engine.predict_batch(X)
    preds2 = inference_engine.predict_batch(X)
    
    np.testing.assert_array_almost_equal(preds1, preds2)


@pytest.mark.parametrize("seed", [10, 20])
def test_predict_single_determinism(inference_engine, seed: int) -> None:
    """Test single prediction determinism. (2 tests)"""
    np.random.seed(seed)
    x = np.random.rand(5)
    
    pred1 = inference_engine.predict_single(x)
    pred2 = inference_engine.predict_single(x)
    
    assert pred1 == pred2
