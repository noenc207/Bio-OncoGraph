import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class PipelineResult:
    """Represents the result of a pipeline execution."""
    status: str
    metrics: Dict[str, float]
    predictions: np.ndarray

class MockPipeline:
    """Mock end-to-end pipeline for testing."""
    def run(self, inputs: Dict[str, np.ndarray]) -> PipelineResult:
        """Runs the mock pipeline with the given inputs."""
        if not inputs:
            raise ValueError("Empty inputs")
        if "data" not in inputs:
            raise KeyError("Missing 'data' key")
        data = inputs["data"]
        if data.size == 0:
            raise ValueError("Empty data array")
        if np.any(np.isnan(data)):
            raise ValueError("NaNs in data")
        
        preds = np.clip(data * 0.8, 0, 1)
        return PipelineResult(
            status="success",
            metrics={"accuracy": 0.95, "loss": 0.05},
            predictions=preds
        )

@pytest.fixture
def pipeline() -> MockPipeline:
    """Fixture providing a mock pipeline instance."""
    return MockPipeline()

def test_pipeline_valid_input_1(pipeline: MockPipeline) -> None:
    """Test pipeline with standard valid input."""
    res = pipeline.run({"data": np.array([0.1, 0.5, 0.9])})
    assert res.status == "success"

def test_pipeline_valid_input_2(pipeline: MockPipeline) -> None:
    """Test pipeline with zero array input."""
    res = pipeline.run({"data": np.zeros(10)})
    assert np.all(res.predictions == 0)

def test_pipeline_valid_input_3(pipeline: MockPipeline) -> None:
    """Test pipeline with ones array input."""
    res = pipeline.run({"data": np.ones(5)})
    assert np.allclose(res.predictions, 0.8)

def test_pipeline_valid_input_4(pipeline: MockPipeline) -> None:
    """Test pipeline metrics structure."""
    res = pipeline.run({"data": np.array([0.5])})
    assert "accuracy" in res.metrics

def test_pipeline_valid_input_5(pipeline: MockPipeline) -> None:
    """Test pipeline metrics loss structure."""
    res = pipeline.run({"data": np.array([0.5])})
    assert "loss" in res.metrics

def test_pipeline_empty_dict(pipeline: MockPipeline) -> None:
    """Test pipeline with empty input dict."""
    with pytest.raises(ValueError, match="Empty inputs"):
        pipeline.run({})

def test_pipeline_missing_data_key(pipeline: MockPipeline) -> None:
    """Test pipeline missing the 'data' key."""
    with pytest.raises(KeyError, match="Missing 'data' key"):
        pipeline.run({"wrong_key": np.array([1, 2])})

def test_pipeline_empty_array(pipeline: MockPipeline) -> None:
    """Test pipeline with empty numpy array."""
    with pytest.raises(ValueError, match="Empty data array"):
        pipeline.run({"data": np.array([])})

def test_pipeline_nan_array(pipeline: MockPipeline) -> None:
    """Test pipeline with NaNs in array."""
    with pytest.raises(ValueError, match="NaNs in data"):
        pipeline.run({"data": np.array([np.nan, 1.0])})

def test_pipeline_clipping_upper(pipeline: MockPipeline) -> None:
    """Test that pipeline predictions are clipped at 1."""
    res = pipeline.run({"data": np.array([2.0, 3.0])})
    assert np.all(res.predictions <= 1.0)

def test_pipeline_clipping_lower(pipeline: MockPipeline) -> None:
    """Test that pipeline predictions are clipped at 0."""
    res = pipeline.run({"data": np.array([-1.0, -2.0])})
    assert np.all(res.predictions >= 0.0)

def test_pipeline_large_input(pipeline: MockPipeline) -> None:
    """Test pipeline with large input array."""
    data = np.random.rand(1000)
    res = pipeline.run({"data": data})
    assert res.predictions.shape == (1000,)

def test_pipeline_2d_input(pipeline: MockPipeline) -> None:
    """Test pipeline with 2D input array."""
    data = np.random.rand(10, 10)
    res = pipeline.run({"data": data})
    assert res.predictions.shape == (10, 10)

def test_pipeline_3d_input(pipeline: MockPipeline) -> None:
    """Test pipeline with 3D input array."""
    data = np.random.rand(5, 5, 5)
    res = pipeline.run({"data": data})
    assert res.predictions.shape == (5, 5, 5)

def test_pipeline_return_type(pipeline: MockPipeline) -> None:
    """Test pipeline returns correct dataclass."""
    res = pipeline.run({"data": np.array([0.5])})
    assert isinstance(res, PipelineResult)

def test_pipeline_metrics_type(pipeline: MockPipeline) -> None:
    """Test pipeline metrics dict value types."""
    res = pipeline.run({"data": np.array([0.5])})
    assert all(isinstance(v, float) for v in res.metrics.values())

def test_pipeline_predictions_type(pipeline: MockPipeline) -> None:
    """Test pipeline predictions array type."""
    res = pipeline.run({"data": np.array([0.5])})
    assert isinstance(res.predictions, np.ndarray)

def test_pipeline_metrics_count(pipeline: MockPipeline) -> None:
    """Test pipeline metrics count."""
    res = pipeline.run({"data": np.array([0.5])})
    assert len(res.metrics) == 2

def test_pipeline_status_success(pipeline: MockPipeline) -> None:
    """Test pipeline status string."""
    res = pipeline.run({"data": np.array([0.5])})
    assert res.status == "success"

def test_pipeline_int_input(pipeline: MockPipeline) -> None:
    """Test pipeline handles int array input."""
    res = pipeline.run({"data": np.array([1, 2, 3])})
    assert res.predictions.dtype == np.float64 or res.predictions.dtype == int

def test_pipeline_dict_mutation(pipeline: MockPipeline) -> None:
    """Test pipeline does not mutate input dict."""
    d = {"data": np.array([0.5])}
    d_copy = dict(d)
    pipeline.run(d)
    assert d.keys() == d_copy.keys()

def test_pipeline_predictions_dtype(pipeline: MockPipeline) -> None:
    """Test pipeline predictions are floats."""
    res = pipeline.run({"data": np.array([0.5, 0.6])})
    assert np.issubdtype(res.predictions.dtype, np.floating)
