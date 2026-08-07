"""
Unit tests for IDataset (via BaseDataset) and MemoryCache.

Covers: dataset length, getitem, metadata, validation, duplicate IDs,
cache hit/miss/stats/clear/disable, transform application on access,
out-of-bounds indexing, missing target warning, invalid metadata.
"""

from typing import Any, Dict, List

import pytest

from bio_oncograph.core.data import DataSample, DatasetMetadata
from bio_oncograph.core.transforms import Identity, LambdaTransform
from bio_oncograph.infrastructure.datasets.base_dataset import (
    BaseDataset,
    MemoryCache,
)


# ── Helpers ─────────────────────────────────────────────────────────

_META = DatasetMetadata(
    name="TestDS",
    version="0.1",
    modality="test",
    label_schema={"event": "bool"},
    num_classes=2,
    source="unit_tests",
)


def _make_samples(n: int = 5) -> List[DataSample]:
    return [
        DataSample(
            sample_id=f"s{i}",
            input=f"input_{i}",
            target=i % 2,
        )
        for i in range(n)
    ]


# ── MemoryCache ─────────────────────────────────────────────────────


def test_cache_put_get() -> None:
    """put/get round-trips a sample."""
    cache = MemoryCache()
    s = _make_samples(1)[0]
    cache.put(0, s)
    assert cache.get(0) is s


def test_cache_miss() -> None:
    """get returns None on miss."""
    cache = MemoryCache()
    assert cache.get(99) is None


def test_cache_disabled() -> None:
    """Disabled cache never stores."""
    cache = MemoryCache(enabled=False)
    s = _make_samples(1)[0]
    cache.put(0, s)
    assert cache.get(0) is None


def test_cache_stats() -> None:
    """Stats track hits and misses."""
    cache = MemoryCache()
    s = _make_samples(1)[0]
    cache.put(0, s)
    cache.get(0)  # hit
    cache.get(1)  # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1


def test_cache_clear() -> None:
    """clear resets cache and counters."""
    cache = MemoryCache()
    cache.put(0, _make_samples(1)[0])
    cache.get(0)
    cache.clear()
    assert cache.stats() == {"hits": 0, "misses": 0, "size": 0}


# ── BaseDataset ─────────────────────────────────────────────────────


def test_dataset_length() -> None:
    """__len__ returns sample count."""
    ds = BaseDataset(_make_samples(10), _META)
    assert len(ds) == 10


def test_dataset_getitem() -> None:
    """__getitem__ returns correct sample."""
    samples = _make_samples(5)
    ds = BaseDataset(samples, _META)
    assert ds[0].sample_id == "s0"
    assert ds[4].sample_id == "s4"


def test_dataset_out_of_bounds() -> None:
    """Out-of-bounds index raises IndexError."""
    ds = BaseDataset(_make_samples(3), _META)
    with pytest.raises(IndexError):
        ds[10]


def test_dataset_negative_index_raises() -> None:
    """Negative index raises IndexError."""
    ds = BaseDataset(_make_samples(3), _META)
    with pytest.raises(IndexError):
        ds[-1]


def test_dataset_metadata() -> None:
    """metadata() returns the descriptor."""
    ds = BaseDataset(_make_samples(3), _META)
    assert ds.metadata().name == "TestDS"
    assert ds.metadata().num_classes == 2


def test_dataset_validate_duplicate_id_raises() -> None:
    """Duplicate sample IDs raise ValueError."""
    samples = [
        DataSample(sample_id="dup", input="a", target=0),
        DataSample(sample_id="dup", input="b", target=1),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        BaseDataset(samples, _META)


def test_dataset_validate_empty_name_raises() -> None:
    """Empty metadata name raises ValueError."""
    bad_meta = DatasetMetadata(
        name="", version="1", modality="m",
        label_schema={}, num_classes=2, source="s",
    )
    with pytest.raises(ValueError, match="name"):
        BaseDataset(_make_samples(1), bad_meta)


def test_dataset_validate_zero_classes_raises() -> None:
    """num_classes < 1 raises ValueError."""
    bad_meta = DatasetMetadata(
        name="X", version="1", modality="m",
        label_schema={}, num_classes=0, source="s",
    )
    with pytest.raises(ValueError, match="num_classes"):
        BaseDataset(_make_samples(1), bad_meta)


def test_dataset_transform_applied() -> None:
    """Transform is applied on __getitem__."""
    def upper_input(s: DataSample) -> DataSample:
        return DataSample(
            sample_id=s.sample_id,
            input=s.input.upper(),
            target=s.target,
            metadata=s.metadata,
        )

    ds = BaseDataset(
        _make_samples(3), _META,
        transform=LambdaTransform(upper_input),
    )
    assert ds[0].input == "INPUT_0"


def test_dataset_cache_works() -> None:
    """Second access returns cached sample (hit)."""
    ds = BaseDataset(_make_samples(3), _META, cache_enabled=True)
    _ = ds[0]  # miss
    _ = ds[0]  # hit
    stats = ds.cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_dataset_cache_disabled() -> None:
    """With cache_enabled=False, no caching occurs."""
    ds = BaseDataset(_make_samples(3), _META, cache_enabled=False)
    _ = ds[0]
    _ = ds[0]
    stats = ds.cache_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2


def test_dataset_clear_cache() -> None:
    """clear_cache resets the cache."""
    ds = BaseDataset(_make_samples(3), _META)
    _ = ds[0]
    ds.clear_cache()
    assert ds.cache_stats()["size"] == 0
