"""Tests for FederatedServer."""
import pytest
import numpy as np
from typing import List

class FederatedServer:
    """Federated Learning Server mock."""
    def __init__(self, initial_model: List[np.ndarray]) -> None:
        self.global_model = initial_model
        
    def get_global_model(self) -> List[np.ndarray]:
        """Get the global model."""
        return self.global_model
        
    def update_model(self, new_model: List[np.ndarray]) -> None:
        """Update the global model."""
        if len(new_model) != len(self.global_model):
            raise ValueError("Mismatched layers")
        self.global_model = new_model
        
    def broadcast(self) -> List[np.ndarray]:
        """Broadcast the model."""
        return self.get_global_model()

@pytest.mark.parametrize("num_layers", [1, 3, 5])
@pytest.mark.parametrize("layer_shape", [(10,), (5, 5), (100, 10)])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_server_init_and_get(num_layers: int, layer_shape: tuple, seed: int) -> None:
    """Test server initialization."""
    np.random.seed(seed)
    initial_model = [np.random.rand(*layer_shape) for _ in range(num_layers)]
    server = FederatedServer(initial_model=initial_model)
    model = server.get_global_model()
    assert len(model) == num_layers
    for i in range(num_layers):
        assert np.array_equal(model[i], initial_model[i])

@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize("layer_shape", [(8,), (4, 4)])
def test_server_update_model_valid(num_layers: int, layer_shape: tuple) -> None:
    """Test valid update."""
    initial = [np.zeros(layer_shape) for _ in range(num_layers)]
    server = FederatedServer(initial)
    new_model = [np.ones(layer_shape) for _ in range(num_layers)]
    server.update_model(new_model)
    updated = server.get_global_model()
    assert np.array_equal(updated[0], np.ones(layer_shape))

@pytest.mark.parametrize("init_layers, new_layers", [(2, 3), (5, 1)])
@pytest.mark.parametrize("layer_shape", [(10,)])
def test_server_update_mismatch(init_layers: int, new_layers: int, layer_shape: tuple) -> None:
    """Test invalid update."""
    server = FederatedServer([np.zeros(layer_shape) for _ in range(init_layers)])
    with pytest.raises(ValueError):
        server.update_model([np.ones(layer_shape) for _ in range(new_layers)])
        
@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("layer_shape", [(5,), (3, 3)])
def test_server_broadcast(num_layers: int, layer_shape: tuple) -> None:
    """Test broadcast."""
    initial = [np.zeros(layer_shape) for _ in range(num_layers)]
    server = FederatedServer(initial)
    broadcasted = server.broadcast()
    assert len(broadcasted) == num_layers
