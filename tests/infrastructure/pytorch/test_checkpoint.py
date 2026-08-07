"""Tests for Checkpointing functionality."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from typing import Dict, Any

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int) -> None:
    """Dummy checkpoint saver."""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(state, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler.LRScheduler = None, strict: bool = True) -> Dict[str, Any]:
    """Dummy checkpoint loader."""
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'], strict=strict)
    if optimizer and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in state:
        scheduler.load_state_dict(state['scheduler_state_dict'])
    return state

@pytest.fixture
def checkpoint_components() -> tuple:
    """Fixture to provide components for checkpointing."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    epoch = 5
    return model, optimizer, scheduler, epoch

def test_save_checkpoint_creates_file(checkpoint_components: tuple) -> None:
    """Test saving checkpoint creates a file on disk."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        assert os.path.exists(path)

def test_load_checkpoint_restores_model_state(checkpoint_components: tuple) -> None:
    """Test loading checkpoint restores model state correctly."""
    model, optimizer, scheduler, epoch = checkpoint_components
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        new_model = nn.Linear(10, 5)
        load_checkpoint(path, new_model)
        
        assert torch.equal(model.weight, new_model.weight)
        assert torch.equal(model.bias, new_model.bias)

def test_load_checkpoint_restores_optimizer_state(checkpoint_components: tuple) -> None:
    """Test loading checkpoint restores optimizer state correctly."""
    model, optimizer, scheduler, epoch = checkpoint_components
    # Update optimizer state by stepping
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
        load_checkpoint(path, new_model, new_optimizer)
        
        assert optimizer.state_dict() == new_optimizer.state_dict()

def test_load_checkpoint_restores_scheduler_state(checkpoint_components: tuple) -> None:
    """Test loading checkpoint restores scheduler state correctly."""
    model, optimizer, scheduler, epoch = checkpoint_components
    scheduler.step()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)
        
        load_checkpoint(path, new_model, new_optimizer, new_scheduler)
        
        assert scheduler.state_dict() == new_scheduler.state_dict()

def test_load_checkpoint_returns_epoch(checkpoint_components: tuple) -> None:
    """Test loading checkpoint returns extra metadata like epoch."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        new_model = nn.Linear(10, 5)
        state = load_checkpoint(path, new_model)
        
        assert state['epoch'] == epoch

def test_load_checkpoint_strict_true_raises_error(checkpoint_components: tuple) -> None:
    """Test strict loading raises error on mismatch."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        # Mismatched model
        new_model = nn.Linear(10, 6)
        with pytest.raises(RuntimeError):
            load_checkpoint(path, new_model, strict=True)

def test_load_checkpoint_strict_false_allows_mismatch(checkpoint_components: tuple) -> None:
    """Test non-strict loading allows missing/unexpected keys."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        class ExtendedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 10))
                self.bias = nn.Parameter(torch.randn(5))
                self.extra = nn.Parameter(torch.randn(2, 2))
                
        new_model = ExtendedModel()
        # Should not raise
        load_checkpoint(path, new_model, strict=False)

def test_save_checkpoint_overwrites_existing(checkpoint_components: tuple) -> None:
    """Test saving checkpoint overwrites existing file."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        # Create dummy file
        with open(path, 'w') as f:
            f.write("dummy")
            
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        state = torch.load(path)
        assert 'model_state_dict' in state

def test_load_checkpoint_missing_file_raises_error() -> None:
    """Test loading from non-existent file raises FileNotFoundError."""
    model = nn.Linear(10, 5)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_file_xyz123.pt", model)

def test_checkpoint_cpu_mapping(checkpoint_components: tuple) -> None:
    """Test that map_location='cpu' handles device transfers safely."""
    model, optimizer, scheduler, epoch = checkpoint_components
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(path, model, optimizer, scheduler, epoch)
        
        new_model = nn.Linear(10, 5)
        # load_checkpoint internally uses map_location='cpu'
        load_checkpoint(path, new_model)
        assert next(new_model.parameters()).device.type == 'cpu'
