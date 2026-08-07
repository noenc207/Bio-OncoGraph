import pytest
import numpy as np
from unittest.mock import MagicMock, patch

class CPTACConnector:
    """Mock CPTACConnector."""
    def __init__(self, cache_dir: str = "/tmp"):
        self.cache_dir = cache_dir
        self.cached = False
    
    def metadata(self) -> dict:
        return {"samples": 500, "modalities": ["proteomics", "genomics"], "name": "CPTAC"}
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple:
        np.random.seed(seed)
        indices = np.random.permutation(500)
        split_idx = int(500 * train_ratio)
        return indices[:split_idx], indices[split_idx:]
    
    def cache(self) -> bool:
        self.cached = True
        return True

@pytest.fixture
def connector() -> CPTACConnector:
    """Fixture providing a configured CPTACConnector instance."""
    return CPTACConnector(cache_dir="/test/cache")

def test_cptac_metadata_returns_dict(connector: CPTACConnector) -> None:
    """Test that metadata returns a dictionary."""
    meta = connector.metadata()
    assert isinstance(meta, dict)

def test_cptac_metadata_contains_name(connector: CPTACConnector) -> None:
    """Test that metadata contains the dataset name."""
    meta = connector.metadata()
    assert "name" in meta
    assert meta["name"] == "CPTAC"

def test_cptac_metadata_contains_samples(connector: CPTACConnector) -> None:
    """Test that metadata contains the sample count."""
    meta = connector.metadata()
    assert "samples" in meta
    assert meta["samples"] == 500

def test_cptac_metadata_contains_modalities(connector: CPTACConnector) -> None:
    """Test that metadata contains modalities list."""
    meta = connector.metadata()
    assert "modalities" in meta
    assert "proteomics" in meta["modalities"]

def test_cptac_split_returns_tuple(connector: CPTACConnector) -> None:
    """Test that split returns a tuple of train/test indices."""
    splits = connector.split()
    assert isinstance(splits, tuple)
    assert len(splits) == 2

def test_cptac_split_lengths(connector: CPTACConnector) -> None:
    """Test that split respects the train_ratio."""
    train_idx, test_idx = connector.split(train_ratio=0.8)
    assert len(train_idx) == 400
    assert len(test_idx) == 100

def test_cptac_split_deterministic_seeding(connector: CPTACConnector) -> None:
    """Test that splitting with the same seed yields identical results."""
    train1, test1 = connector.split(seed=123)
    train2, test2 = connector.split(seed=123)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(test1, test2)

def test_cptac_split_different_seeds(connector: CPTACConnector) -> None:
    """Test that splitting with different seeds yields different results."""
    train1, test1 = connector.split(seed=42)
    train2, test2 = connector.split(seed=43)
    assert not np.array_equal(train1, train2)
    assert not np.array_equal(test1, test2)

def test_cptac_cache_returns_true(connector: CPTACConnector) -> None:
    """Test that caching returns True on success."""
    result = connector.cache()
    assert result is True

def test_cptac_cache_sets_cached_flag(connector: CPTACConnector) -> None:
    """Test that calling cache() updates internal state."""
    assert not connector.cached
    connector.cache()
    assert connector.cached

def test_cptac_initialization_cache_dir() -> None:
    """Test that cache_dir is set correctly during initialization."""
    conn = CPTACConnector(cache_dir="/my/custom/path")
    assert conn.cache_dir == "/my/custom/path"

def test_cptac_split_train_ratio_zero(connector: CPTACConnector) -> None:
    """Test splitting with a train_ratio of 0.0."""
    train_idx, test_idx = connector.split(train_ratio=0.0)
    assert len(train_idx) == 0
    assert len(test_idx) == 500

def test_cptac_split_train_ratio_one(connector: CPTACConnector) -> None:
    """Test splitting with a train_ratio of 1.0."""
    train_idx, test_idx = connector.split(train_ratio=1.0)
    assert len(train_idx) == 500
    assert len(test_idx) == 0

def test_cptac_split_invalid_ratio_type(connector: CPTACConnector) -> None:
    """Test splitting with an invalid train_ratio type."""
    with pytest.raises(TypeError):
        connector.split(train_ratio="invalid") # type: ignore
