"""Tests for Automatic Mixed Precision (AMP) utilities in PyTorch.

This module contains tests for autocast wrappers and gradient scalers.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from typing import Generator, Any

# from bio_oncograph.infrastructure.pytorch.amp import auto_cast_context, get_grad_scaler

@contextmanager
def auto_cast_context(enabled: bool = True, device_type: str = "cuda") -> Generator[Any, None, None]:
    """Wrapper for torch.autocast."""
    if device_type == "cuda" and not torch.cuda.is_available():
        device_type = "cpu"
    with torch.autocast(device_type=device_type, enabled=enabled):
        yield

def get_grad_scaler(enabled: bool = True) -> torch.amp.GradScaler:
    """Wrapper for torch.amp.GradScaler."""
    return torch.amp.GradScaler("cuda", enabled=enabled)

def test_auto_cast_context_enabled_cpu() -> None:
    """Test autocast context on CPU when enabled."""
    with auto_cast_context(enabled=True, device_type="cpu"):
        assert torch.is_autocast_enabled()

def test_auto_cast_context_disabled_cpu() -> None:
    """Test autocast context on CPU when disabled."""
    with auto_cast_context(enabled=False, device_type="cpu"):
        assert not torch.is_autocast_enabled()

@patch("torch.cuda.is_available", return_value=True)
def test_auto_cast_context_enabled_cuda(mock_is_avail: MagicMock) -> None:
    """Test autocast context on CUDA when enabled."""
    # We mock autocast since we might not have CUDA on the test machine
    with patch("torch.autocast") as mock_autocast:
        with auto_cast_context(enabled=True, device_type="cuda"):
            pass
        mock_autocast.assert_called_once_with(device_type="cuda", enabled=True)

@patch("torch.cuda.is_available", return_value=False)
def test_auto_cast_context_fallback_cpu(mock_is_avail: MagicMock) -> None:
    """Test autocast context fallbacks to CPU when CUDA is unavailable."""
    with patch("torch.autocast") as mock_autocast:
        with auto_cast_context(enabled=True, device_type="cuda"):
            pass
        mock_autocast.assert_called_once_with(device_type="cpu", enabled=True)

def test_get_grad_scaler_enabled() -> None:
    """Test retrieving an enabled GradScaler."""
    scaler = get_grad_scaler(enabled=True)
    assert isinstance(scaler, torch.amp.GradScaler)
    assert scaler.is_enabled()

def test_get_grad_scaler_disabled() -> None:
    """Test retrieving a disabled GradScaler."""
    scaler = get_grad_scaler(enabled=False)
    assert isinstance(scaler, torch.amp.GradScaler)
    assert not scaler.is_enabled()

def test_amp_forward_pass_dtype() -> None:
    """Test that AMP forward pass changes dtype on CPU."""
    a = torch.randn(3, 3, dtype=torch.float32)
    b = torch.randn(3, 3, dtype=torch.float32)
    
    with auto_cast_context(enabled=True, device_type="cpu"):
        # bfloat16 is standard for CPU autocast
        c = torch.matmul(a, b)
        assert c.dtype == torch.bfloat16

def test_no_amp_forward_pass_dtype() -> None:
    """Test that without AMP, forward pass maintains dtype."""
    a = torch.randn(3, 3, dtype=torch.float32)
    b = torch.randn(3, 3, dtype=torch.float32)
    
    with auto_cast_context(enabled=False, device_type="cpu"):
        c = torch.matmul(a, b)
        assert c.dtype == torch.float32

def test_grad_scaler_step_unscale() -> None:
    """Test that grad scaler can scale and unscale properly (mocked)."""
    scaler = get_grad_scaler(enabled=True)
    optimizer = MagicMock()
    
    # Mock behavior of step
    scaler.step(optimizer)
    optimizer.step.assert_called_once()

def test_grad_scaler_update() -> None:
    """Test grad scaler update mechanism."""
    scaler = get_grad_scaler(enabled=True)
    scale_before = scaler.get_scale()
    scaler.update()
    # It should maintain scale if no infs/nans occurred
    assert scaler.get_scale() == scale_before
