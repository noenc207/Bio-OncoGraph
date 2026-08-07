"""Tests for PipelineOrchestrator.

Validates the PipelineOrchestrator logic by passing arrays through a mocked component pipeline.
"""

from typing import Any, List, Callable
import pytest
import numpy as np

class PipelineOrchestrator:
    """Mock PipelineOrchestrator."""
    def __init__(self) -> None:
        self.pipeline: List[Callable[[np.ndarray], np.ndarray]] = []
        
    def add_component(self, component: Callable[[np.ndarray], np.ndarray]) -> None:
        self.pipeline.append(component)
        
    def run(self, input_data: np.ndarray) -> np.ndarray:
        data = input_data
        for comp in self.pipeline:
            data = comp(data)
        return data

@pytest.fixture
def orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator()

def multiply_by_two(x: np.ndarray) -> np.ndarray:
    return x * 2.0

def add_five(x: np.ndarray) -> np.ndarray:
    return x + 5.0

def square(x: np.ndarray) -> np.ndarray:
    return x ** 2

@pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4), (100, 100)])
@pytest.mark.parametrize("multiplier", [1.0, 2.0, -1.0, 0.0])
def test_single_component_pipeline(orchestrator: PipelineOrchestrator, shape: tuple, multiplier: float) -> None:
    """Test pipeline with a single component."""
    orchestrator.add_component(lambda x: x * multiplier)
    data = np.ones(shape)
    result = orchestrator.run(data)
    expected = np.ones(shape) * multiplier
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("shape", [(10, 10), (1, 1), (50,)])
@pytest.mark.parametrize("num_components", [2, 3, 5, 10])
def test_multi_component_pipeline(orchestrator: PipelineOrchestrator, shape: tuple, num_components: int) -> None:
    """Test pipeline with multiple components sequentially applied."""
    for _ in range(num_components):
        orchestrator.add_component(add_five)
        
    data = np.zeros(shape)
    result = orchestrator.run(data)
    expected = np.zeros(shape) + (5.0 * num_components)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("pipeline_funcs", [
    ([multiply_by_two, add_five], lambda x: (x * 2) + 5),
    ([add_five, multiply_by_two], lambda x: (x + 5) * 2),
    ([square, multiply_by_two, add_five], lambda x: (x**2 * 2) + 5),
])
@pytest.mark.parametrize("input_val", [0.0, 1.0, -2.5, 10.0])
def test_pipeline_order(orchestrator: PipelineOrchestrator, pipeline_funcs: tuple, input_val: float) -> None:
    """Test if pipeline components are applied in the correct order."""
    funcs, expected_func = pipeline_funcs
    for f in funcs:
        orchestrator.add_component(f)
        
    data = np.array([input_val])
    result = orchestrator.run(data)
    expected = expected_func(data)
    np.testing.assert_array_equal(result, expected)
