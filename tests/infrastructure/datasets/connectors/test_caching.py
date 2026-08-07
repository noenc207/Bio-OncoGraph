"""Tests for Caching functionality."""

import pytest
import os
import json
import tempfile
from typing import Any, Optional

class CacheManager:
    """Manages local caching for dataset connectors."""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        """Save data to cache."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, key: str) -> Optional[Any]:
        """Load data from cache."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache."""
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def clear_all(self) -> None:
        """Clear the entire cache directory."""
        for file in os.listdir(self.cache_dir):
            if file.endswith(".json"):
                os.remove(os.path.join(self.cache_dir, file))

@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def cache_manager(cache_dir):
    return CacheManager(cache_dir)

def test_cache_init_creates_dir(cache_dir):
    """Test that CacheManager creates the directory if it doesn't exist."""
    new_dir = os.path.join(cache_dir, "new_cache")
    CacheManager(new_dir)
    assert os.path.exists(new_dir)

def test_cache_save_and_load(cache_manager):
    """Test saving and loading data."""
    data = {"key": "value", "num": 42}
    cache_manager.save("test_key", data)
    loaded = cache_manager.load("test_key")
    assert loaded == data

def test_cache_load_missing(cache_manager):
    """Test loading a missing key returns None."""
    assert cache_manager.load("missing_key") is None

def test_cache_invalidate_existing(cache_manager):
    """Test invalidating an existing key."""
    cache_manager.save("test_key", [1, 2, 3])
    assert cache_manager.invalidate("test_key") is True
    assert cache_manager.load("test_key") is None

def test_cache_invalidate_missing(cache_manager):
    """Test invalidating a missing key returns False."""
    assert cache_manager.invalidate("missing_key") is False

def test_cache_overwrite(cache_manager):
    """Test saving to the same key overwrites previous data."""
    cache_manager.save("key", "val1")
    cache_manager.save("key", "val2")
    assert cache_manager.load("key") == "val2"

def test_cache_clear_all(cache_manager):
    """Test clearing all cache entries."""
    cache_manager.save("k1", 1)
    cache_manager.save("k2", 2)
    cache_manager.clear_all()
    assert cache_manager.load("k1") is None
    assert cache_manager.load("k2") is None

def test_cache_save_complex_types(cache_manager):
    """Test saving lists and nested dicts."""
    data = {"a": [1, 2, {"b": 3}]}
    cache_manager.save("complex", data)
    assert cache_manager.load("complex") == data

def test_cache_large_data(cache_manager):
    """Test saving and loading a large list."""
    data = list(range(10000))
    cache_manager.save("large", data)
    assert cache_manager.load("large") == data

def test_cache_invalid_type(cache_manager):
    """Test saving un-serializable type raises TypeError."""
    class Dummy: pass
    with pytest.raises(TypeError):
        cache_manager.save("dummy", Dummy())

def test_cache_clear_empty(cache_manager):
    """Test clear_all on empty cache doesn't raise error."""
    cache_manager.clear_all() # Should not raise
    assert True

def test_cache_file_format(cache_dir, cache_manager):
    """Test that the file is actually saved as JSON."""
    cache_manager.save("test", {"a": 1})
    with open(os.path.join(cache_dir, "test.json"), "r") as f:
        content = f.read()
    assert '"a": 1' in content
