"""Caching infrastructure for WSI patches and metadata."""

import collections
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseCache:
    """Base LRU cache implementation."""
    
    def __init__(self, capacity: int = 1000) -> None:
        """Initialize cache with capacity.
        
        Args:
            capacity: Maximum number of items to store.
        """
        self._capacity = capacity
        self._cache: collections.OrderedDict[Any, Any] = collections.OrderedDict()
        
    def get(self, key: Any) -> Optional[Any]:
        """Get an item from the cache.
        
        Args:
            key: Cache key.
            
        Returns:
            The cached value, or None if not found.
        """
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]
        
    def put(self, key: Any, value: Any) -> None:
        """Put an item into the cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
            
    def invalidate(self, key: Any) -> None:
        """Invalidate a specific cache key.
        
        Args:
            key: Key to remove.
        """
        if key in self._cache:
            del self._cache[key]
            
    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()
        
    def size(self) -> int:
        """Get the current number of cached items.
        
        Returns:
            Number of items.
        """
        return len(self._cache)


class PatchCache(BaseCache):
    """LRU Cache for extracted patches.
    
    Key: (slide_id, x, y, level, patch_width, patch_height)
    Value: PatchResult or np.ndarray
    """
    
    def get(self, key: Tuple[str, int, int, int, int, int]) -> Optional[Any]:
        return super().get(key)
        
    def put(self, key: Tuple[str, int, int, int, int, int], value: Any) -> None:
        super().put(key, value)


class ThumbnailCache(BaseCache):
    """Cache for slide thumbnails.
    
    Key: (slide_id, width, height)
    """
    pass


class MetadataCache(BaseCache):
    """Cache for slide metadata.
    
    Key: slide_id
    """
    pass
