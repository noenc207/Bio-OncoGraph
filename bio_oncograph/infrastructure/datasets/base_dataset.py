"""
Concrete base dataset with caching support.

Provides:
  - ``MemoryCache`` — in-memory sample cache with hit/miss stats.
  - ``BaseDataset`` — concrete ``IDataset`` implementation that
    stores samples in a list, applies transforms on access, and
    optionally caches transformed samples.

This module lives in the Infrastructure layer and may import from
``core/``, but never from ``application/``.

Design references:
  - Software Architecture § 6 (Data Layer)
  - Software Architecture § 7 (Caching)
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from bio_oncograph.core.data import DataSample, DatasetMetadata
from bio_oncograph.core.dataset import IDataset
from bio_oncograph.core.transforms import Transform

logger = logging.getLogger("BioOncoGraph")


# ── MemoryCache ─────────────────────────────────────────────────────


class MemoryCache:
    """
    In-memory LRU-style cache for ``DataSample`` instances.

    Tracks hit and miss counts for observability.  Cache can be
    enabled or disabled at construction time.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialise the cache.

        Args:
            enabled: If ``False``, ``get`` always returns ``None``
                and ``put`` is a no-op.
        """
        self._cache: Dict[int, DataSample] = {}
        self._hits: int = 0
        self._misses: int = 0
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Return whether caching is enabled."""
        return self._enabled

    def get(self, key: int) -> Optional[DataSample]:
        """
        Retrieve a cached sample.

        Args:
            key: The sample index.

        Returns:
            The cached ``DataSample``, or ``None`` on miss or if
            caching is disabled.
        """
        if not self._enabled:
            self._misses += 1
            return None
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: int, value: DataSample) -> None:
        """
        Store a sample in the cache.

        Args:
            key: The sample index.
            value: The ``DataSample`` to cache.
        """
        if self._enabled:
            self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, int]:
        """
        Return cache performance statistics.

        Returns:
            Dict with ``hits``, ``misses``, and ``size`` keys.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def __repr__(self) -> str:
        return (
            f"MemoryCache(enabled={self._enabled}, "
            f"size={len(self._cache)})"
        )


# ── BaseDataset ─────────────────────────────────────────────────────


class BaseDataset(IDataset):
    """
    Concrete dataset backed by an in-memory sample list.

    Features:
      - Optional ``Transform`` applied on each ``__getitem__``.
      - Optional ``MemoryCache`` for transformed samples.
      - Fail-fast validation (duplicate IDs, metadata).

    Args:
        samples: Ordered sequence of ``DataSample`` instances.
        dataset_metadata: Immutable dataset descriptor.
        transform: Optional transform applied on access.
        cache_enabled: Whether to cache transformed samples.
    """

    def __init__(
        self,
        samples: Sequence[DataSample],
        dataset_metadata: DatasetMetadata,
        transform: Optional[Transform] = None,
        cache_enabled: bool = True,
    ) -> None:
        """Initialise and validate the dataset."""
        self._samples: List[DataSample] = list(samples)
        self._metadata_obj = dataset_metadata
        self._transform = transform
        self._cache = MemoryCache(enabled=cache_enabled)

        self.validate()
        logger.info(
            "BaseDataset '%s' created: %d samples, "
            "cache=%s",
            dataset_metadata.name,
            len(self._samples),
            cache_enabled,
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)

    def __getitem__(self, index: int) -> DataSample:
        """
        Retrieve a sample, applying transforms and caching.

        Args:
            index: Zero-based sample index.

        Returns:
            The (possibly transformed and cached) ``DataSample``.

        Raises:
            IndexError: If *index* is out of bounds.
        """
        if index < 0 or index >= len(self._samples):
            raise IndexError(
                f"Index {index} out of range "
                f"[0, {len(self._samples)})."
            )

        cached = self._cache.get(index)
        if cached is not None:
            return cached

        sample = self._samples[index]
        if self._transform is not None:
            sample = self._transform.apply(sample)

        self._cache.put(index, sample)
        return sample

    def metadata(self) -> DatasetMetadata:
        """Return the dataset metadata."""
        return self._metadata_obj

    def validate(self) -> None:
        """
        Validate dataset integrity.

        Checks:
          - No duplicate sample IDs.
          - Metadata name is non-empty.
          - num_classes >= 1.
          - No sample has ``None`` target (warning only).

        Raises:
            ValueError: On duplicate IDs or invalid metadata.
        """
        # Duplicate IDs
        ids = [s.sample_id for s in self._samples]
        counts = Counter(ids)
        dupes = [sid for sid, cnt in counts.items() if cnt > 1]
        if dupes:
            raise ValueError(
                f"Duplicate sample IDs detected: {dupes}"
            )

        # Metadata checks
        if not self._metadata_obj.name:
            raise ValueError(
                "Dataset metadata 'name' must not be empty."
            )
        if self._metadata_obj.num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1, "
                f"got {self._metadata_obj.num_classes}."
            )

        # Missing target (warning, not error)
        missing = [
            s.sample_id
            for s in self._samples
            if s.target is None
        ]
        if missing:
            logger.warning(
                "%d samples have None targets: %s",
                len(missing),
                missing[:5],
            )

    def cache_stats(self) -> Dict[str, int]:
        """Return cache performance statistics."""
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Clear the sample cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"BaseDataset(name={self._metadata_obj.name!r}, "
            f"samples={len(self._samples)})"
        )
