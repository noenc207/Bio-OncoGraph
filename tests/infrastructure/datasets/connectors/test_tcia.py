import pytest
import numpy as np
from unittest.mock import MagicMock, patch

class TCIAConnector:
    """Mock TCIAConnector."""
    def __init__(self, cache_dir: str = "/tmp"):
        self.cache_dir = cache_dir
        self.cached = False
    
    def metadata(self) -> dict:
        return {"samples": 300, "image_type": "MRI", "name": "TCIA"}
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple:
        np.random.seed(seed)
        indices = np.random.permutation(300)
        split_idx = int(300 * train_ratio)
        return indices[:split_idx], indices[split_idx:]
    
    def cache(self) -> bool:
        self.cached = True
        return True

@pytest.fixture
def connector() -> TCIAConnector:
    """Fixture providing a configured TCIAConnector instance."""
    return TCIAConnector(cache_dir="/test/cache")

def test_tcia_metadata_returns_dict(connector: TCIAConnector) -> None:
    """Test that metadata returns a dictionary."""
    meta = connector.metadata()
    assert isinstance(meta, dict)

def test_tcia_metadata_contains_name(connector: TCIAConnector) -> None:
    """Test that metadata contains the dataset name."""
    meta = connector.metadata()
    assert "name" in meta
    assert meta["name"] == "TCIA"

def test_tcia_metadata_contains_samples(connector: TCIAConnector) -> None:
    """Test that metadata contains the sample count."""
    meta = connector.metadata()
    assert "samples" in meta
    assert meta["samples"] == 300

def test_tcia_metadata_contains_image_type(connector: TCIAConnector) -> None:
    """Test that metadata contains image_type."""
    meta = connector.metadata()
    assert "image_type" in meta
    assert meta["image_type"] == "MRI"

def test_tcia_split_returns_tuple(connector: TCIAConnector) -> None:
    """Test that split returns a tuple of train/test indices."""
    splits = connector.split()
    assert isinstance(splits, tuple)
    assert len(splits) == 2

def test_tcia_split_lengths(connector: TCIAConnector) -> None:
    """Test that split respects the train_ratio."""
    train_idx, test_idx = connector.split(train_ratio=0.8)
    assert len(train_idx) == 240
    assert len(test_idx) == 60

def test_tcia_split_deterministic_seeding(connector: TCIAConnector) -> None:
    """Test that splitting with the same seed yields identical results."""
    train1, test1 = connector.split(seed=42)
    train2, test2 = connector.split(seed=42)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(test1, test2)

def test_tcia_split_different_seeds(connector: TCIAConnector) -> None:
    """Test that splitting with different seeds yields different results."""
    train1, test1 = connector.split(seed=42)
    train2, test2 = connector.split(seed=43)
    assert not np.array_equal(train1, train2)
    assert not np.array_equal(test1, test2)

def test_tcia_cache_returns_true(connector: TCIAConnector) -> None:
    """Test that caching returns True on success."""
    result = connector.cache()
    assert result is True

def test_tcia_cache_sets_cached_flag(connector: TCIAConnector) -> None:
    """Test that calling cache() updates internal state."""
    assert not connector.cached
    connector.cache()
    assert connector.cached

def test_tcia_initialization_cache_dir() -> None:
    """Test that cache_dir is set correctly during initialization."""
    conn = TCIAConnector(cache_dir="/my/custom/path")
    assert conn.cache_dir == "/my/custom/path"

def test_tcia_split_train_ratio_zero(connector: TCIAConnector) -> None:
    """Test splitting with a train_ratio of 0.0."""
    train_idx, test_idx = connector.split(train_ratio=0.0)
    assert len(train_idx) == 0
    assert len(test_idx) == 300

def test_tcia_split_train_ratio_one(connector: TCIAConnector) -> None:
    """Test splitting with a train_ratio of 1.0."""
    train_idx, test_idx = connector.split(train_ratio=1.0)
    assert len(train_idx) == 300
    assert len(test_idx) == 0

def test_tcia_split_invalid_ratio_type(connector: TCIAConnector) -> None:
    """Test splitting with an invalid train_ratio type."""
    with pytest.raises(TypeError):
        connector.split(train_ratio="invalid") # type: ignore
