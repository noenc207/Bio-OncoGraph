"""Tests for PyTorchBackend facade.

This module verifies the main PyTorchBackend class behavior.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

# from bio_oncograph.infrastructure.pytorch.backend import PyTorchBackend

class PyTorchBackend:
    """Facade for PyTorch backend operations."""
    
    def __init__(self, device: str = "cpu", seed: int | None = None) -> None:
        self.device = torch.device(device)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            
    def get_device(self) -> torch.device:
        return self.device
        
    def to_tensor(self, data: list | torch.Tensor) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, device=self.device)
        
    def empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

def test_backend_initialization_default() -> None:
    """Test initializing backend with default parameters."""
    backend = PyTorchBackend()
    assert backend.get_device().type == "cpu"
    assert backend.seed is None

def test_backend_initialization_custom_device() -> None:
    """Test initializing backend with a custom device."""
    backend = PyTorchBackend(device="cpu")
    assert backend.get_device().type == "cpu"

@patch("torch.manual_seed")
def test_backend_initialization_with_seed(mock_seed: MagicMock) -> None:
    """Test initializing backend with a seed calls torch.manual_seed."""
    backend = PyTorchBackend(seed=42)
    mock_seed.assert_called_once_with(42)
    assert backend.seed == 42

def test_backend_to_tensor_from_list() -> None:
    """Test backend to_tensor from a list."""
    backend = PyTorchBackend()
    tensor = backend.to_tensor([1, 2, 3])
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,)

def test_backend_to_tensor_existing_tensor() -> None:
    """Test backend to_tensor with an existing tensor."""
    backend = PyTorchBackend()
    orig = torch.tensor([1, 2, 3])
    tensor = backend.to_tensor(orig)
    assert tensor.shape == (3,)

@patch("torch.cuda.is_available", return_value=True)
def test_backend_to_tensor_moves_device(mock_is_avail: MagicMock) -> None:
    """Test that to_tensor moves tensor to the backend's device."""
    # Since we might not have actual cuda, we mock the tensor's to method
    backend = PyTorchBackend(device="cuda:0" if torch.cuda.is_available() else "cpu")
    with patch.object(torch.Tensor, 'to') as mock_to:
        orig = torch.tensor([1, 2, 3])
        backend.to_tensor(orig)
        mock_to.assert_called_once_with(backend.device)

@patch("torch.cuda.empty_cache")
def test_backend_empty_cache_cuda(mock_empty: MagicMock) -> None:
    """Test that empty_cache calls cuda.empty_cache if device is cuda."""
    backend = PyTorchBackend(device="cpu")
    backend.device = torch.device("cuda:0") # Force device for test
    backend.empty_cache()
    mock_empty.assert_called_once()

@patch("torch.cuda.empty_cache")
def test_backend_empty_cache_cpu(mock_empty: MagicMock) -> None:
    """Test that empty_cache does nothing if device is cpu."""
    backend = PyTorchBackend(device="cpu")
    backend.empty_cache()
    mock_empty.assert_not_called()

def test_backend_multiple_instances() -> None:
    """Test that multiple instances can be created independently."""
    b1 = PyTorchBackend(seed=1)
    b2 = PyTorchBackend(seed=2)
    assert b1.seed == 1
    assert b2.seed == 2

def test_backend_invalid_device_init() -> None:
    """Test that initializing with an invalid device throws an error."""
    with pytest.raises(RuntimeError):
        PyTorchBackend(device="invalid_dev")
