"""Caching mechanisms for foundation model embeddings."""

import logging
from typing import Dict, Optional

from bio_oncograph.research.foundation_models.embeddings import EmbeddingContainer


logger = logging.getLogger(__name__)


class FeatureCache:
    """In-memory cache for features/embeddings."""

    def __init__(self) -> None:
        """Initializes the cache."""
        self._cache: Dict[str, EmbeddingContainer] = {}

    def put(self, key: str, embedding: EmbeddingContainer) -> None:
        """Stores an embedding in the cache.

        Args:
            key: Cache key.
            embedding: The embedding container to cache.
        """
        self._cache[key] = embedding
        logger.info(f"Stored item in cache with key: {key}")

    def get(self, key: str) -> Optional[EmbeddingContainer]:
        """Retrieves an embedding from the cache.

        Args:
            key: Cache key.

        Returns:
            The embedding container if found, else None.
        """
        if key in self._cache:
            logger.info(f"Cache hit for key: {key}")
            return self._cache[key]
        logger.info(f"Cache miss for key: {key}")
        return None

    def invalidate(self, key: str) -> None:
        """Removes an item from the cache.

        Args:
            key: Cache key.
        """
        if key in self._cache:
            del self._cache[key]
            logger.info(f"Invalidated cache key: {key}")

    def clear(self) -> None:
        """Clears all items from the cache."""
        self._cache.clear()
        logger.info("Cleared the feature cache.")
