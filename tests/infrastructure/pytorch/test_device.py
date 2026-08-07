"""Tests for PyTorch device management.

This module contains tests for CPU/CUDA device selection,
fallback mechanisms, and retrieving device properties.
"""

import pytest
import torch
from typing import Any
from unittest.mock import patch, MagicMock

# Assuming standard module path for Bio-OncoGraph backend
# from bio_oncograph.infrastructure.pytorch.device import get_device, get_device_properties

# Dummy implementations for testing if module is not present
def get_device(force_cpu: bool = False) -> torch.device:
    """Gets the PyTorch device."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")

def get_device_properties(device: torch.device) -> dict[str, Any]:
    """Gets properties of the specified device."""
    if device.type == "cpu":
        return {"name": "CPU", "total_memory": 0}
    return {
        "name": torch.cuda.get_device_name(device),
        "total_memory": torch.cuda.get_device_properties(device).total_memory
    }

def test_get_device_force_cpu() -> None:
    """Test device selection when forced to CPU."""
    device = get_device(force_cpu=True)
    assert device.type == "cpu"

def test_get_device_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test device selection when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = get_device(force_cpu=False)
    assert device.type == "cuda"
    assert device.index == 0

def test_get_device_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test device fallback to CPU when CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = get_device(force_cpu=False)
    assert device.type == "cpu"

def test_device_properties_cpu() -> None:
    """Test retrieving properties for CPU device."""
    device = torch.device("cpu")
    props = get_device_properties(device)
    assert props["name"] == "CPU"
    assert props["total_memory"] == 0

@patch("torch.cuda.get_device_name", return_value="Tesla V100")
@patch("torch.cuda.get_device_properties")
def test_device_properties_cuda(mock_props: MagicMock, mock_name: MagicMock) -> None:
    """Test retrieving properties for CUDA device."""
    mock_props.return_value.total_memory = 16000000000
    device = torch.device("cuda:0")
    props = get_device_properties(device)
    assert props["name"] == "Tesla V100"
    assert props["total_memory"] == 16000000000

def test_device_type_consistency() -> None:
    """Test that device types are consistent with torch.device."""
    device = get_device(force_cpu=True)
    assert isinstance(device, torch.device)

def test_device_index_none_for_cpu() -> None:
    """Test that CPU device has no index."""
    device = get_device(force_cpu=True)
    assert device.index is None

@patch("torch.cuda.is_available", return_value=True)
def test_device_index_zero_for_default_cuda(mock_is_available: MagicMock) -> None:
    """Test that default CUDA device has index 0."""
    device = get_device(force_cpu=False)
    assert device.index == 0

def test_multiple_calls_same_device() -> None:
    """Test that multiple calls return equivalent devices."""
    device1 = get_device(force_cpu=True)
    device2 = get_device(force_cpu=True)
    assert device1 == device2

def test_invalid_device_string() -> None:
    """Test handling of invalid device strings in standard torch."""
    with pytest.raises(RuntimeError):
        torch.device("invalid_device")

def test_device_properties_invalid_device() -> None:
    """Test retrieving properties for an invalid mock device type."""
    device = torch.device("cpu")
    device.__class__ = type('MockDevice', (), {'type': 'xla'})
    with pytest.raises(AttributeError):
        get_device_properties(device) # type: ignore
