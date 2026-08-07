import pytest
import numpy as np
from typing import Dict, Any, List

class ScientificValidator:
    """Validates scientific arrays."""
    def validate_bounds(self, array: np.ndarray) -> bool:
        """Ensures array contains no NaNs, Infs, and is within [0, 1]."""
        if np.any(np.isnan(array)):
            return False
        if np.any(np.isinf(array)):
            return False
        if np.any(array < 0) or np.any(array > 1):
            return False
        return True

class EndToEndPipeline:
    """Mock sequential pipeline."""
    def __init__(self, validator: ScientificValidator):
        """Initializes pipeline with a validator."""
        self.validator = validator
        self.state = "initialized"
    
    def step1(self) -> None:
        """Executes step 1."""
        self.state = "step1_done"
        
    def step2(self, data: np.ndarray) -> np.ndarray:
        """Executes step 2."""
        if self.state != "step1_done":
            raise RuntimeError("Out of order")
        if not self.validator.validate_bounds(data):
            raise ValueError("Invalid bounds")
        self.state = "step2_done"
        return data * 0.5

@pytest.fixture
def validator() -> ScientificValidator:
    """Fixture providing a ScientificValidator."""
    return ScientificValidator()

@pytest.fixture
def pipeline(validator: ScientificValidator) -> EndToEndPipeline:
    """Fixture providing an EndToEndPipeline."""
    return EndToEndPipeline(validator)

def test_validator_valid_bounds(validator: ScientificValidator) -> None:
    """Test validator with strictly valid array."""
    assert validator.validate_bounds(np.array([0.1, 0.5, 0.9]))

def test_validator_lower_bound(validator: ScientificValidator) -> None:
    """Test validator with exactly 0."""
    assert validator.validate_bounds(np.array([0.0, 0.5]))

def test_validator_upper_bound(validator: ScientificValidator) -> None:
    """Test validator with exactly 1."""
    assert validator.validate_bounds(np.array([0.5, 1.0]))

def test_validator_out_of_bounds_high(validator: ScientificValidator) -> None:
    """Test validator with value > 1."""
    assert not validator.validate_bounds(np.array([1.1]))

def test_validator_out_of_bounds_low(validator: ScientificValidator) -> None:
    """Test validator with value < 0."""
    assert not validator.validate_bounds(np.array([-0.1]))

def test_validator_nan(validator: ScientificValidator) -> None:
    """Test validator with NaN."""
    assert not validator.validate_bounds(np.array([np.nan]))

def test_validator_inf(validator: ScientificValidator) -> None:
    """Test validator with Inf."""
    assert not validator.validate_bounds(np.array([np.inf]))

def test_validator_neg_inf(validator: ScientificValidator) -> None:
    """Test validator with negative Inf."""
    assert not validator.validate_bounds(np.array([-np.inf]))

def test_pipeline_init(pipeline: EndToEndPipeline) -> None:
    """Test pipeline initial state."""
    assert pipeline.state == "initialized"

def test_pipeline_step1(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 1 state transition."""
    pipeline.step1()
    assert pipeline.state == "step1_done"

def test_pipeline_step2_out_of_order(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 raises error if called before step 1."""
    with pytest.raises(RuntimeError, match="Out of order"):
        pipeline.step2(np.array([0.5]))

def test_pipeline_step2_invalid_bounds(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 raises error if data invalid."""
    pipeline.step1()
    with pytest.raises(ValueError, match="Invalid bounds"):
        pipeline.step2(np.array([1.5]))

def test_pipeline_step2_success(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 success."""
    pipeline.step1()
    res = pipeline.step2(np.array([0.4, 0.8]))
    assert np.allclose(res, [0.2, 0.4])
    assert pipeline.state == "step2_done"

def test_pipeline_step2_zeros(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 with zeros."""
    pipeline.step1()
    res = pipeline.step2(np.zeros(5))
    assert np.allclose(res, 0.0)

def test_pipeline_step2_ones(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 with ones."""
    pipeline.step1()
    res = pipeline.step2(np.ones(5))
    assert np.allclose(res, 0.5)

def test_pipeline_step2_shape(pipeline: EndToEndPipeline) -> None:
    """Test pipeline step 2 maintains shape."""
    pipeline.step1()
    data = np.random.rand(3, 3)
    res = pipeline.step2(data)
    assert res.shape == (3, 3)
