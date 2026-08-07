import pytest
from typing import Any, Dict

class BaseCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._keys: list = []

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._keys.remove(key)
        elif len(self._cache) >= self.max_size:
            lru = self._keys.pop(0)
            del self._cache[lru]
        self._cache[key] = value
        self._keys.append(key)

    def get(self, key: str) -> Any:
        if key in self._cache:
            self._keys.remove(key)
            self._keys.append(key)
            return self._cache[key]
        return None

    def invalidate(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._keys.remove(key)

    def clear(self) -> None:
        self._cache.clear()
        self._keys.clear()

    def size(self) -> int:
        return len(self._cache)

class PatchCache(BaseCache):
    pass

class ThumbnailCache(BaseCache):
    pass

class MetadataCache(BaseCache):
    pass

@pytest.fixture
def cache() -> BaseCache:
    """Provide a small cache for testing."""
    return BaseCache(max_size=3)

# 16 tests
def test_put_and_get(cache: BaseCache) -> None:
    """Test basic put and get."""
    cache.put("k1", "v1")
    assert cache.get("k1") == "v1"

def test_get_missing(cache: BaseCache) -> None:
    """Test get missing key."""
    assert cache.get("k2") is None

def test_invalidate(cache: BaseCache) -> None:
    """Test invalidation."""
    cache.put("k1", "v1")
    cache.invalidate("k1")
    assert cache.get("k1") is None

def test_invalidate_missing(cache: BaseCache) -> None:
    """Test invalidating missing key."""
    cache.invalidate("k1")
    assert cache.size() == 0

def test_clear(cache: BaseCache) -> None:
    """Test clear cache."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.clear()
    assert cache.size() == 0

def test_size_initial(cache: BaseCache) -> None:
    """Test initial size."""
    assert cache.size() == 0

def test_size_after_put(cache: BaseCache) -> None:
    """Test size after puts."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    assert cache.size() == 2

def test_lru_eviction(cache: BaseCache) -> None:
    """Test LRU eviction."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")
    cache.put("k4", "v4")  # evicts k1
    assert cache.size() == 3
    assert cache.get("k1") is None
    assert cache.get("k4") == "v4"

def test_lru_update_on_get(cache: BaseCache) -> None:
    """Test LRU updates on get."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")
    cache.get("k1")  # k1 is now most recently used
    cache.put("k4", "v4")  # evicts k2
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1"

def test_lru_update_on_put(cache: BaseCache) -> None:
    """Test LRU updates on put."""
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")
    cache.put("k1", "v1_new")  # k1 is updated and MRU
    cache.put("k4", "v4")  # evicts k2
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1_new"

def test_patch_cache_instance() -> None:
    """Test PatchCache instantiation."""
    c = PatchCache()
    c.put("p1", "data")
    assert c.size() == 1

def test_thumbnail_cache_instance() -> None:
    """Test ThumbnailCache instantiation."""
    c = ThumbnailCache()
    c.put("t1", "thumb")
    assert c.size() == 1

def test_metadata_cache_instance() -> None:
    """Test MetadataCache instantiation."""
    c = MetadataCache()
    c.put("m1", "meta")
    assert c.size() == 1

def test_cache_zero_size() -> None:
    """Test cache with zero size."""
    c = BaseCache(max_size=0)
    c.put("k1", "v1")
    assert c.size() == 0

def test_cache_large_inserts(cache: BaseCache) -> None:
    """Test many inserts."""
    for i in range(10):
        cache.put(f"k{i}", i)
    assert cache.size() == 3
    assert cache.get("k9") == 9

def test_clear_empty(cache: BaseCache) -> None:
    """Test clear on empty cache."""
    cache.clear()
    assert cache.size() == 0
