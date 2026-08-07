"""Tests for feature caching mechanisms."""
import pytest
from typing import Any

class FeatureCache:
    """Simple LRU feature cache."""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}
        self._order = []
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)
        
    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key not in self._cache:
            raise KeyError(f"Key {key} not found in cache")
        return self._cache[key]
        
    def invalidate(self, key: str) -> None:
        """Invalidate single key."""
        if key in self._cache:
            del self._cache[key]
            self._order.remove(key)
            
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._order.clear()


@pytest.mark.parametrize("key", [f"key_{i}" for i in range(15)])
@pytest.mark.parametrize("value", [1, "test", [1, 2, 3], {"a": 1}, 3.14])
def test_feature_cache_put_get(key: str, value: Any) -> None:
    """Test basic put and get operations on FeatureCache."""
    cache = FeatureCache()
    cache.put(key, value)
    assert cache.get(key) == value

@pytest.mark.parametrize("missing_key", ["nonexistent1", "nonexistent2", "missing", "void", "null"])
@pytest.mark.parametrize("repeat", list(range(3)))
def test_feature_cache_get_missing(missing_key: str, repeat: int) -> None:
    """Test getting missing keys raises KeyError."""
    cache = FeatureCache()
    with pytest.raises(KeyError):
        cache.get(missing_key)

@pytest.mark.parametrize("max_size", [1, 2, 5, 10])
@pytest.mark.parametrize("extra_items", [1, 3, 5, 10])
def test_feature_cache_eviction(max_size: int, extra_items: int) -> None:
    """Test that max size evictions remove the oldest items."""
    cache = FeatureCache(max_size=max_size)
    total_items = max_size + extra_items
    for i in range(total_items):
        cache.put(f"k_{i}", i)
    
    # The first 'extra_items' should be evicted
    for i in range(extra_items):
        with pytest.raises(KeyError):
            cache.get(f"k_{i}")
            
    # The remaining 'max_size' items should be present
    for i in range(extra_items, total_items):
        assert cache.get(f"k_{i}") == i

@pytest.mark.parametrize("num_items", [5, 10, 20, 50])
@pytest.mark.parametrize("repeat", list(range(3)))
def test_feature_cache_invalidate_clear(num_items: int, repeat: int) -> None:
    """Test invalidation of single items and clearing the entire cache."""
    cache = FeatureCache(max_size=100)
    for i in range(num_items):
        cache.put(f"key_{i}", i)
    
    # Invalidate one item
    cache.invalidate("key_0")
    with pytest.raises(KeyError):
        cache.get("key_0")
        
    # Clear all items
    cache.clear()
    for i in range(1, num_items):
        with pytest.raises(KeyError):
            cache.get(f"key_{i}")
