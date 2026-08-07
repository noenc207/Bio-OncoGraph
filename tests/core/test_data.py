"""
Unit tests for core data types and DataSplitter.

Covers: DataSample creation/immutability/hashability, DatasetMetadata,
SplitResult, FoldIndices, DataSplitter.split (random & stratified),
DataSplitter.train_split/validation_split/test_split,
DataSplitter.k_fold_split, determinism, reproducibility, edge cases.
"""

from typing import Any

import pytest

from bio_oncograph.core.data import (
    DataSample,
    DatasetMetadata,
    DataSplitter,
    FoldIndices,
    SplitResult,
)


# ── DataSample ──────────────────────────────────────────────────────


def test_sample_creation() -> None:
    """DataSample can be created with required fields."""
    s = DataSample(sample_id="s1", input="x", target=1)
    assert s.sample_id == "s1"
    assert s.input == "x"
    assert s.target == 1
    assert s.metadata == {}


def test_sample_immutability() -> None:
    """Frozen dataclass prevents attribute reassignment."""
    s = DataSample(sample_id="s1", input="x", target=1)
    with pytest.raises(AttributeError):
        s.sample_id = "s2"  # type: ignore[misc]


def test_sample_hashable() -> None:
    """DataSample is hashable by sample_id."""
    s1 = DataSample(sample_id="s1", input="a", target=1)
    s2 = DataSample(sample_id="s1", input="b", target=2)
    assert hash(s1) == hash(s2)


def test_sample_equality_by_id() -> None:
    """Two samples with same ID are equal."""
    s1 = DataSample(sample_id="s1", input="a", target=1)
    s2 = DataSample(sample_id="s1", input="b", target=2)
    assert s1 == s2


def test_sample_inequality() -> None:
    """Different IDs means not equal."""
    s1 = DataSample(sample_id="s1", input="a", target=1)
    s2 = DataSample(sample_id="s2", input="a", target=1)
    assert s1 != s2


def test_sample_in_set() -> None:
    """DataSample works in sets (hashable + eq)."""
    s1 = DataSample(sample_id="s1", input="a", target=1)
    s2 = DataSample(sample_id="s1", input="b", target=2)
    s3 = DataSample(sample_id="s2", input="c", target=3)
    assert len({s1, s2, s3}) == 2  # s1 and s2 collapse


def test_sample_metadata() -> None:
    """Metadata dict is stored."""
    s = DataSample(
        sample_id="s1", input="x", target=1,
        metadata={"source": "tcga"},
    )
    assert s.metadata["source"] == "tcga"


# ── DatasetMetadata ─────────────────────────────────────────────────


def test_metadata_creation() -> None:
    """DatasetMetadata can be created."""
    m = DatasetMetadata(
        name="TCGA", version="1.0", modality="histopathology",
        label_schema={"event": "int"}, num_classes=4,
        source="gdc.cancer.gov",
    )
    assert m.name == "TCGA"
    assert m.num_classes == 4


def test_metadata_immutable() -> None:
    """DatasetMetadata is frozen."""
    m = DatasetMetadata(
        name="X", version="1", modality="m",
        label_schema={}, num_classes=2, source="s",
    )
    with pytest.raises(AttributeError):
        m.name = "Y"  # type: ignore[misc]


def test_metadata_hashable() -> None:
    """DatasetMetadata is hashable."""
    m = DatasetMetadata(
        name="X", version="1", modality="m",
        label_schema={}, num_classes=2, source="s",
    )
    assert isinstance(hash(m), int)


# ── SplitResult / FoldIndices ───────────────────────────────────────


def test_split_result_frozen() -> None:
    """SplitResult is immutable."""
    sr = SplitResult(
        train_indices=(0, 1),
        val_indices=(2,),
        test_indices=(3,),
    )
    with pytest.raises(AttributeError):
        sr.train_indices = (5,)  # type: ignore[misc]


def test_fold_indices_frozen() -> None:
    """FoldIndices is immutable."""
    fi = FoldIndices(train_indices=(0, 1), val_indices=(2,))
    with pytest.raises(AttributeError):
        fi.val_indices = (99,)  # type: ignore[misc]


# ── DataSplitter.split ──────────────────────────────────────────────


def test_split_sizes() -> None:
    """Split produces correct approximate sizes."""
    result = DataSplitter.split(100, 0.7, 0.15, 0.15, seed=42)
    total = (
        len(result.train_indices)
        + len(result.val_indices)
        + len(result.test_indices)
    )
    assert total == 100
    assert len(result.train_indices) == 70
    assert len(result.val_indices) == 15


def test_split_no_overlap() -> None:
    """Train, val, test have no overlapping indices."""
    result = DataSplitter.split(50, 0.6, 0.2, 0.2, seed=7)
    all_indices = set(result.train_indices) | set(result.val_indices) | set(result.test_indices)
    assert len(all_indices) == 50


def test_split_deterministic() -> None:
    """Same seed produces identical splits."""
    r1 = DataSplitter.split(100, seed=42)
    r2 = DataSplitter.split(100, seed=42)
    assert r1 == r2


def test_split_different_seeds() -> None:
    """Different seeds produce different splits."""
    r1 = DataSplitter.split(100, seed=42)
    r2 = DataSplitter.split(100, seed=99)
    assert r1.train_indices != r2.train_indices


def test_split_sorted() -> None:
    """Returned indices are sorted."""
    result = DataSplitter.split(100, seed=42)
    assert list(result.train_indices) == sorted(result.train_indices)
    assert list(result.val_indices) == sorted(result.val_indices)
    assert list(result.test_indices) == sorted(result.test_indices)


def test_split_invalid_ratios_raises() -> None:
    """Ratios not summing to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataSplitter.split(100, 0.5, 0.5, 0.5)


def test_split_negative_ratio_raises() -> None:
    """Negative ratio raises ValueError."""
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        DataSplitter.split(100, -0.1, 0.6, 0.5)


def test_split_zero_samples_raises() -> None:
    """Zero samples raises ValueError."""
    with pytest.raises(ValueError, match="num_samples"):
        DataSplitter.split(0)


# ── Stratified split ───────────────────────────────────────────────


def test_stratified_split() -> None:
    """Stratified split preserves approximate label balance."""
    labels = [0] * 50 + [1] * 50
    result = DataSplitter.split(
        100, 0.7, 0.15, 0.15, seed=42, labels=labels,
    )
    total = (
        len(result.train_indices)
        + len(result.val_indices)
        + len(result.test_indices)
    )
    assert total == 100

    # Check label 0 is in both train and test
    train_labels = [labels[i] for i in result.train_indices]
    assert 0 in train_labels
    assert 1 in train_labels


# ── Convenience methods ────────────────────────────────────────────


def test_train_split() -> None:
    """train_split returns training indices only."""
    indices = DataSplitter.train_split(100, seed=42)
    full = DataSplitter.split(100, seed=42)
    assert indices == full.train_indices


def test_validation_split() -> None:
    """validation_split returns validation indices only."""
    indices = DataSplitter.validation_split(100, seed=42)
    full = DataSplitter.split(100, seed=42)
    assert indices == full.val_indices


def test_test_split() -> None:
    """test_split returns test indices only."""
    indices = DataSplitter.test_split(100, seed=42)
    full = DataSplitter.split(100, seed=42)
    assert indices == full.test_indices


# ── K-fold ──────────────────────────────────────────────────────────


def test_kfold_num_folds() -> None:
    """k_fold_split returns exactly k folds."""
    folds = DataSplitter.k_fold_split(100, k=5, seed=42)
    assert len(folds) == 5


def test_kfold_no_overlap_in_fold() -> None:
    """Train and val within a fold have no overlap."""
    folds = DataSplitter.k_fold_split(100, k=5, seed=42)
    for fold in folds:
        train_set = set(fold.train_indices)
        val_set = set(fold.val_indices)
        assert train_set & val_set == set()


def test_kfold_all_indices_covered() -> None:
    """Union of train+val covers all indices."""
    folds = DataSplitter.k_fold_split(50, k=5, seed=42)
    for fold in folds:
        all_idx = set(fold.train_indices) | set(fold.val_indices)
        assert all_idx == set(range(50))


def test_kfold_deterministic() -> None:
    """Same seed produces same folds."""
    f1 = DataSplitter.k_fold_split(100, k=3, seed=42)
    f2 = DataSplitter.k_fold_split(100, k=3, seed=42)
    for a, b in zip(f1, f2):
        assert a == b


def test_kfold_k_less_than_2_raises() -> None:
    """k < 2 raises ValueError."""
    with pytest.raises(ValueError, match="k must be >= 2"):
        DataSplitter.k_fold_split(100, k=1)


def test_kfold_more_folds_than_samples_raises() -> None:
    """k > num_samples raises ValueError."""
    with pytest.raises(ValueError, match="num_samples"):
        DataSplitter.k_fold_split(3, k=5)
