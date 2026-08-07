"""Tests for Custom Dataset, DataLoader, and collate_fn."""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict

class DummyDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, size: int = 100) -> None:
        """Initialize dataset with given size."""
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        if idx >= self.size or idx < -self.size:
            raise IndexError("Index out of bounds")
        return self.data[idx], self.labels[idx]

def dummy_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dummy collate function."""
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {"features": data, "targets": labels}

def test_dataset_length() -> None:
    """Test dataset returns correct length."""
    dataset = DummyDataset(50)
    assert len(dataset) == 50

def test_dataset_getitem() -> None:
    """Test dataset returns correct item format."""
    dataset = DummyDataset(10)
    x, y = dataset[0]
    assert x.shape == (10,)
    assert y.shape == ()
    assert y.dtype == torch.int64

def test_dataset_out_of_bounds() -> None:
    """Test dataset raises IndexError for out of bounds."""
    dataset = DummyDataset(10)
    with pytest.raises(IndexError):
        _ = dataset[10]

def test_dataloader_batch_size() -> None:
    """Test DataLoader returns batches of correct size."""
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=16)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (16, 10)
    assert batch_y.shape == (16,)

def test_dataloader_drop_last() -> None:
    """Test DataLoader drop_last functionality."""
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=30, drop_last=True)
    assert len(loader) == 3

def test_dataloader_shuffle() -> None:
    """Test DataLoader shuffle functionality."""
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=100, shuffle=True)
    batch_x, _ = next(iter(loader))
    
    # It's highly unlikely that shuffled data is exactly the same as sequential
    assert not torch.equal(batch_x, dataset.data)

def test_dataloader_collate_fn() -> None:
    """Test custom collate_fn processing."""
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=16, collate_fn=dummy_collate_fn)
    batch = next(iter(loader))
    
    assert isinstance(batch, dict)
    assert "features" in batch
    assert "targets" in batch
    assert batch["features"].shape == (16, 10)
    assert batch["targets"].shape == (16,)

def test_dataloader_pinned_memory() -> None:
    """Test DataLoader pin_memory setting."""
    dataset = DummyDataset(100)
    loader = DataLoader(dataset, batch_size=16, pin_memory=True)
    batch_x, batch_y = next(iter(loader))
    
    # If CUDA is not available, pinned memory tensors might not register as pinned
    # But we can at least check it doesn't crash and returns tensors
    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)

def test_dataloader_num_workers() -> None:
    """Test DataLoader with multiple workers."""
    dataset = DummyDataset(100)
    # Use small number of workers to avoid OS limits in CI
    loader = DataLoader(dataset, batch_size=16, num_workers=2)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (16, 10)

def test_dataloader_iteration_completes() -> None:
    """Test complete iteration over the DataLoader."""
    dataset = DummyDataset(50)
    loader = DataLoader(dataset, batch_size=16)
    
    batches = 0
    total_samples = 0
    for x, y in loader:
        batches += 1
        total_samples += x.size(0)
        
    assert batches == 4
    assert total_samples == 50
