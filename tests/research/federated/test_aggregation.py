"""Tests for Aggregation algorithms."""
import pytest
import numpy as np
from typing import List

def fed_avg(weights: List[List[np.ndarray]], client_sizes: List[int]) -> List[np.ndarray]:
    """Federated Averaging."""
    total = sum(client_sizes)
    return [sum(w[i] * c / total for w, c in zip(weights, client_sizes)) for i in range(len(weights[0]))]

def robust_agg(weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Robust Median Aggregation."""
    return [np.median([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]

@pytest.mark.parametrize("num_clients", [2, 5, 10, 20])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("layer_shape", [(5,), (10, 10)])
@pytest.mark.parametrize("sizes", ["uniform", "random"])
def test_fed_avg(num_clients: int, num_layers: int, layer_shape: tuple, sizes: str) -> None:
    """Test FedAvg."""
    weights = [[np.ones(layer_shape) * i for _ in range(num_layers)] for i in range(num_clients)]
    if sizes == "uniform":
        client_sizes = [10] * num_clients
    else:
        client_sizes = [10 * (i + 1) for i in range(num_clients)]
        
    result = fed_avg(weights, client_sizes)
    assert len(result) == num_layers
    
    # manual calculation
    total = sum(client_sizes)
    expected_val = sum(i * c / total for i, c in enumerate(client_sizes))
    assert np.allclose(result[0], np.ones(layer_shape) * expected_val)

@pytest.mark.parametrize("num_clients", [3, 7, 11])
@pytest.mark.parametrize("num_layers", [1, 2, 4])
@pytest.mark.parametrize("layer_shape", [(4,), (8, 8)])
@pytest.mark.parametrize("outlier_idx", [0, 1])
def test_robust_agg(num_clients: int, num_layers: int, layer_shape: tuple, outlier_idx: int) -> None:
    """Test Robust Aggregation (Median)."""
    weights = [[np.ones(layer_shape)] * num_layers for _ in range(num_clients)]
    weights[outlier_idx] = [np.ones(layer_shape) * 1000] * num_layers
    
    result = robust_agg(weights)
    assert np.allclose(result[0], np.ones(layer_shape))

@pytest.mark.parametrize("num_layers_c1, num_layers_c2", [(1, 2), (3, 1)])
def test_agg_mismatch_layers(num_layers_c1: int, num_layers_c2: int) -> None:
    """Test mismatched layer counts."""
    weights = [[np.ones((5,))] * num_layers_c1, [np.ones((5,))] * num_layers_c2]
    with pytest.raises(IndexError):
        fed_avg(weights, [10, 10])
