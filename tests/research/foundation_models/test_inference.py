"""Tests for InferenceEngine.

Contains comprehensive tests for InferenceEngine covering batching,
output normalization, and deterministic random state management.
"""

from typing import Any
import pytest
import numpy as np

class InferenceEngine:
    """Mock InferenceEngine for testing."""
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
    def process_batch(self, data: np.ndarray, batch_size: int) -> list:
        num_batches = int(np.ceil(len(data) / batch_size))
        return [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        
    def normalize_output(self, output: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return output / norms
        
    def generate_random_outputs(self, shape: tuple) -> np.ndarray:
        return self.rng.random(shape)


@pytest.fixture
def inference_engine() -> InferenceEngine:
    return InferenceEngine(seed=123)

@pytest.mark.parametrize("data_size", [10, 50, 105, 1000])
@pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32, 64, 128])
def test_batching(inference_engine: InferenceEngine, data_size: int, batch_size: int) -> None:
    """Test data batching logic."""
    data = np.arange(data_size)
    batches = inference_engine.process_batch(data, batch_size)
    
    expected_batches = int(np.ceil(data_size / batch_size))
    assert len(batches) == expected_batches
    
    # Reconstruct and compare
    reconstructed = np.concatenate(batches)
    np.testing.assert_array_equal(reconstructed, data)

@pytest.mark.parametrize("shape", [(10, 128), (1, 64), (100, 256), (5, 512)])
@pytest.mark.parametrize("scale", [1.0, 0.01, 100.0, 0.0])
def test_output_normalization(inference_engine: InferenceEngine, shape: tuple, scale: float) -> None:
    """Test output normalization to ensure unit length vectors."""
    output = np.random.randn(*shape) * scale
    normalized = inference_engine.normalize_output(output)
    
    norms = np.linalg.norm(normalized, axis=1)
    if scale == 0.0:
        # If all zeros, norm should be close to 0 due to epsilon
        assert np.allclose(norms, 0.0, atol=1e-5)
    else:
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

@pytest.mark.parametrize("seed", [0, 42, 123, 9999, 123456])
@pytest.mark.parametrize("shape", [(10, 10), (50, 5), (2, 2, 2)])
def test_deterministic_random_outputs(seed: int, shape: tuple) -> None:
    """Test if the inference engine produces deterministic outputs given a seed."""
    engine1 = InferenceEngine(seed=seed)
    engine2 = InferenceEngine(seed=seed)
    
    out1 = engine1.generate_random_outputs(shape)
    out2 = engine2.generate_random_outputs(shape)
    
    np.testing.assert_array_equal(out1, out2)

@pytest.mark.parametrize("batch_idx", range(15))
def test_batch_processing_edge_cases(inference_engine: InferenceEngine, batch_idx: int) -> None:
    """Test extreme batch sizing."""
    data = np.zeros((100, 10))
    batches = inference_engine.process_batch(data, 100 + batch_idx)
    assert len(batches) == 1
    assert len(batches[0]) == 100
