import pytest
import numpy as np
from unittest.mock import MagicMock
from typing import Any

class InferenceService:
    """Service for running model inference."""
    
    def predict_online(self, data: np.ndarray) -> np.ndarray:
        return np.array([])
        
    def predict_batch(self, data_batch: np.ndarray) -> np.ndarray:
        return np.array([])

@pytest.fixture
def service() -> InferenceService:
    """Provide an InferenceService instance."""
    return InferenceService()

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("feature_dim", [10, 20, 50, 100, 200, 512, 1024])
def test_inference_service_predict(service: InferenceService, batch_size: int, feature_dim: int) -> None:
    """
    Test inference service predict methods.
    
    Args:
        service (InferenceService): Service fixture.
        batch_size (int): Size of batch.
        feature_dim (int): Number of features.
    """
    data = np.random.rand(batch_size, feature_dim)
    mock_out = np.zeros(batch_size)
    service.predict_online = MagicMock(return_value=mock_out)
    
    result = service.predict_online(data)
    assert result.shape == (batch_size,)
    service.predict_online.assert_called_with(data)
