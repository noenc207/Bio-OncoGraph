"""Tests for serialization of metadata."""

import pytest
import torch
import os
import tempfile
from typing import Dict, Any

def save_metadata(path: str, metadata: Dict[str, Any]) -> None:
    """Dummy function to save metadata to .pt."""
    torch.save(metadata, path)

def load_metadata(path: str) -> Dict[str, Any]:
    """Dummy function to load metadata from .pt."""
    return torch.load(path)

@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Fixture providing sample metadata."""
    return {
        "version": "1.0",
        "description": "Bio-OncoGraph metadata",
        "config": {
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "metrics": torch.tensor([0.95, 0.96, 0.97]),
        "tags": ["cancer", "graph", "pytorch"]
    }

def test_save_metadata_creates_file(sample_metadata: Dict[str, Any]) -> None:
    """Test that saving metadata creates a .pt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.pt")
        save_metadata(path, sample_metadata)
        assert os.path.exists(path)

def test_load_metadata_restores_dict(sample_metadata: Dict[str, Any]) -> None:
    """Test that loaded metadata matches saved metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.pt")
        save_metadata(path, sample_metadata)
        
        loaded = load_metadata(path)
        assert loaded["version"] == "1.0"
        assert loaded["config"]["batch_size"] == 32

def test_serialize_tensors_in_metadata(sample_metadata: Dict[str, Any]) -> None:
    """Test that tensors within metadata are correctly serialized/deserialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.pt")
        save_metadata(path, sample_metadata)
        
        loaded = load_metadata(path)
        assert torch.allclose(loaded["metrics"], sample_metadata["metrics"])

def test_serialize_lists_in_metadata(sample_metadata: Dict[str, Any]) -> None:
    """Test that lists within metadata are correctly preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.pt")
        save_metadata(path, sample_metadata)
        
        loaded = load_metadata(path)
        assert loaded["tags"] == ["cancer", "graph", "pytorch"]

def test_save_metadata_pth_extension(sample_metadata: Dict[str, Any]) -> None:
    """Test saving metadata with .pth extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.pth")
        save_metadata(path, sample_metadata)
        assert os.path.exists(path)
        
        loaded = load_metadata(path)
        assert loaded["version"] == "1.0"

def test_empty_metadata_serialization() -> None:
    """Test serialization of empty metadata dictionary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty_meta.pt")
        save_metadata(path, {})
        loaded = load_metadata(path)
        assert loaded == {}

def test_large_metadata_serialization() -> None:
    """Test serialization of relatively large metadata."""
    large_metadata = {"data": torch.randn(100, 100)}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large_meta.pt")
        save_metadata(path, large_metadata)
        loaded = load_metadata(path)
        assert loaded["data"].shape == (100, 100)

def test_load_metadata_invalid_path() -> None:
    """Test loading from an invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_metadata("invalid_path_12345.pt")

def test_metadata_corrupted_file() -> None:
    """Test loading corrupted file raises RuntimeError or similar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.pt")
        with open(path, 'wb') as f:
            f.write(b"this is not a valid pytorch file format")
            
        with pytest.raises(Exception):
            load_metadata(path)
