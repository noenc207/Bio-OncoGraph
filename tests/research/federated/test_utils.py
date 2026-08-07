import pytest
import numpy as np
from typing import List

def compute_weighted_average(arrays: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if not arrays or not weights:
        raise ValueError("Empty input")
    if sum(weights) == 0:
        raise ValueError("Zero weight sum")
    return np.average(arrays, axis=0, weights=weights)

def add_gaussian_noise(array: np.ndarray, std_dev: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    noise = np.random.normal(0, std_dev, array.shape)
    return array + noise

def simulate_network_delay(base_delay: float, jitter: float) -> float:
    return base_delay + jitter

@pytest.mark.parametrize("num_arrays", [1, 3, 5])
@pytest.mark.parametrize("array_size", [10, 50])
@pytest.mark.parametrize("weight_type", ["uniform", "random", "zero"])
def test_compute_weighted_average(num_arrays: int, array_size: int, weight_type: str) -> None:
    """Test computing weighted averages with different conditions."""
    arrays = [np.ones(array_size) for _ in range(num_arrays)]
    if weight_type == "uniform":
        weights = [1.0 for _ in range(num_arrays)]
    elif weight_type == "random":
        weights = [float(i + 1) for i in range(num_arrays)]
    elif weight_type == "zero":
        weights = [0.0 for _ in range(num_arrays)]
    
    if weight_type == "zero":
        with pytest.raises(ValueError):
            compute_weighted_average(arrays, weights)
    else:
        result = compute_weighted_average(arrays, weights)
        assert result.shape == (array_size,)

def test_compute_weighted_average_empty() -> None:
    """Test empty input handling."""
    with pytest.raises(ValueError):
        compute_weighted_average([], [])

@pytest.mark.parametrize("shape", [(10,), (5, 5)])
@pytest.mark.parametrize("std_dev", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("seed", [42, 123])
def test_add_gaussian_noise(shape: tuple, std_dev: float, seed: int) -> None:
    """Test adding deterministic gaussian noise."""
    array = np.zeros(shape)
    noisy1 = add_gaussian_noise(array, std_dev, seed)
    noisy2 = add_gaussian_noise(array, std_dev, seed)
    assert np.array_equal(noisy1, noisy2)
    assert noisy1.shape == shape

@pytest.mark.parametrize("base", [10.0, 50.0])
@pytest.mark.parametrize("jitter", [1.0, 5.0])
def test_simulate_network_delay(base: float, jitter: float) -> None:
    """Test simple network delay simulation."""
    assert simulate_network_delay(base, jitter) == base + jitter
