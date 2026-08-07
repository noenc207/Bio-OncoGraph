"""
Core data types and splitting utilities for Bio-OncoGraph.

Provides:
  - ``DataSample`` — immutable, hashable data record.
  - ``DatasetMetadata`` — immutable dataset descriptor.
  - ``SplitResult`` / ``FoldIndices`` — immutable split containers.
  - ``DataSplitter`` — deterministic, reproducible data splitting
    with optional stratification and k-fold support.

All types are pure Python — **no** ``torch``, ``numpy``, or
``pandas`` imports — so they can be used safely in the Domain layer.

Design references:
  - Software Architecture § 6 (Data Layer)
  - Software Architecture § 21 (Reproducibility)
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("BioOncoGraph")


# ── DataSample ──────────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class DataSample:
    """
    Immutable, hashable record representing a single data sample.

    Equality and hashing are based on ``sample_id`` only, so two
    samples with the same ID are considered identical regardless of
    their ``input`` or ``target`` payloads.

    Attributes:
        sample_id: Unique identifier for this sample.
        input: Arbitrary input payload (e.g. a feature tensor).
        target: Arbitrary target payload (e.g. a survival label).
        metadata: Optional key-value metadata for this sample.
    """

    sample_id: str
    input: Any
    target: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSample):
            return NotImplemented
        return self.sample_id == other.sample_id

    def __hash__(self) -> int:
        return hash(self.sample_id)

    def __repr__(self) -> str:
        return (
            f"DataSample(sample_id={self.sample_id!r}, "
            f"target={self.target!r})"
        )


# ── DatasetMetadata ─────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetMetadata:
    """
    Immutable descriptor for a dataset.

    Attributes:
        name: Human-readable dataset name.
        version: Semantic version string.
        modality: Data modality (e.g. ``"histopathology"``).
        label_schema: Mapping describing label semantics.
        num_classes: Number of classification/survival classes.
        source: Origin or DOI of the dataset.
    """

    name: str
    version: str
    modality: str
    label_schema: Dict[str, Any]
    num_classes: int
    source: str

    def __hash__(self) -> int:
        """Hash by name and version (excludes mutable dict)."""
        return hash(
            (self.name, self.version, self.modality,
             self.num_classes, self.source)
        )


# ── Split containers ───────────────────────────────────────────────


@dataclass(frozen=True)
class SplitResult:
    """
    Result of a train / validation / test split.

    Indices are stored as sorted tuples for immutability and
    deterministic iteration.
    """

    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]


@dataclass(frozen=True)
class FoldIndices:
    """Train / validation indices for a single k-fold iteration."""

    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]


# ── DataSplitter ────────────────────────────────────────────────────


class DataSplitter:
    """
    Deterministic, reproducible data splitting utilities.

    All methods use ``random.Random(seed)`` for local PRNG state,
    ensuring that results are independent of the global random state
    and fully reproducible across runs.
    """

    # ── Full split ──────────────────────────────────────────────

    @staticmethod
    def split(
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        labels: Optional[Sequence[int]] = None,
    ) -> SplitResult:
        """
        Split sample indices into train / val / test sets.

        Args:
            num_samples: Total number of samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.
            labels: Optional label sequence for stratified splitting.

        Returns:
            A ``SplitResult`` with sorted index tuples.

        Raises:
            ValueError: If ratios do not sum to 1 or are invalid.
        """
        DataSplitter._validate_ratios(
            train_ratio, val_ratio, test_ratio
        )
        if num_samples < 1:
            raise ValueError(
                f"num_samples must be >= 1, got {num_samples}."
            )

        rng = random.Random(seed)

        if labels is not None:
            return DataSplitter._stratified_split(
                num_samples, labels,
                train_ratio, val_ratio, rng,
            )

        indices = list(range(num_samples))
        rng.shuffle(indices)

        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)

        train = tuple(sorted(indices[:n_train]))
        val = tuple(sorted(indices[n_train: n_train + n_val]))
        test = tuple(sorted(indices[n_train + n_val:]))

        logger.info(
            "Split %d samples: train=%d, val=%d, test=%d",
            num_samples, len(train), len(val), len(test),
        )
        return SplitResult(
            train_indices=train,
            val_indices=val,
            test_indices=test,
        )

    # ── Convenience per-split accessors ─────────────────────────

    @staticmethod
    def train_split(
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[int, ...]:
        """
        Return only the training indices.

        Args:
            num_samples: Total number of samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.

        Returns:
            Sorted tuple of training indices.
        """
        result = DataSplitter.split(
            num_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=round(1.0 - train_ratio - val_ratio, 10),
            seed=seed,
        )
        return result.train_indices

    @staticmethod
    def validation_split(
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[int, ...]:
        """
        Return only the validation indices.

        Args:
            num_samples: Total number of samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.

        Returns:
            Sorted tuple of validation indices.
        """
        result = DataSplitter.split(
            num_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=round(1.0 - train_ratio - val_ratio, 10),
            seed=seed,
        )
        return result.val_indices

    @staticmethod
    def test_split(
        num_samples: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[int, ...]:
        """
        Return only the test indices.

        Args:
            num_samples: Total number of samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            seed: Random seed.

        Returns:
            Sorted tuple of test indices.
        """
        result = DataSplitter.split(
            num_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=round(1.0 - train_ratio - val_ratio, 10),
            seed=seed,
        )
        return result.test_indices

    # ── K-fold ──────────────────────────────────────────────────

    @staticmethod
    def k_fold_split(
        num_samples: int,
        k: int = 5,
        seed: int = 42,
        labels: Optional[Sequence[int]] = None,
    ) -> List[FoldIndices]:
        """
        Generate k-fold cross-validation splits.

        Args:
            num_samples: Total number of samples.
            k: Number of folds (must be >= 2).
            seed: Random seed.
            labels: Optional label sequence for stratified folding.

        Returns:
            A list of ``k`` ``FoldIndices`` objects.

        Raises:
            ValueError: If *k* < 2 or *num_samples* < *k*.
        """
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}.")
        if num_samples < k:
            raise ValueError(
                f"num_samples ({num_samples}) must be >= k ({k})."
            )

        rng = random.Random(seed)
        indices = list(range(num_samples))
        rng.shuffle(indices)

        # Distribute indices into k buckets
        buckets: List[List[int]] = [[] for _ in range(k)]
        for i, idx in enumerate(indices):
            buckets[i % k].append(idx)

        folds: List[FoldIndices] = []
        for fold_idx in range(k):
            val_bucket = buckets[fold_idx]
            train_bucket: List[int] = []
            for j in range(k):
                if j != fold_idx:
                    train_bucket.extend(buckets[j])
            folds.append(
                FoldIndices(
                    train_indices=tuple(sorted(train_bucket)),
                    val_indices=tuple(sorted(val_bucket)),
                )
            )

        logger.info(
            "K-fold split: %d samples into %d folds", num_samples, k
        )
        return folds

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _validate_ratios(
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> None:
        """Validate that split ratios are sane."""
        for name, ratio in [
            ("train_ratio", train_ratio),
            ("val_ratio", val_ratio),
            ("test_ratio", test_ratio),
        ]:
            if ratio < 0.0 or ratio > 1.0:
                raise ValueError(
                    f"{name} must be in [0, 1], got {ratio}."
                )
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.6f}."
            )

    @staticmethod
    def _stratified_split(
        num_samples: int,
        labels: Sequence[int],
        train_ratio: float,
        val_ratio: float,
        rng: random.Random,
    ) -> SplitResult:
        """Stratified split preserving label distribution."""
        if len(labels) != num_samples:
            raise ValueError(
                f"len(labels)={len(labels)} != "
                f"num_samples={num_samples}."
            )

        label_to_indices: Dict[int, List[int]] = {}
        for idx in range(num_samples):
            label = labels[idx]
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        train_all: List[int] = []
        val_all: List[int] = []
        test_all: List[int] = []

        for label in sorted(label_to_indices.keys()):
            group = label_to_indices[label]
            rng.shuffle(group)
            n = len(group)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))

            train_all.extend(group[:n_train])
            val_all.extend(group[n_train: n_train + n_val])
            test_all.extend(group[n_train + n_val:])

        logger.info(
            "Stratified split: train=%d, val=%d, test=%d",
            len(train_all), len(val_all), len(test_all),
        )
        return SplitResult(
            train_indices=tuple(sorted(train_all)),
            val_indices=tuple(sorted(val_all)),
            test_indices=tuple(sorted(test_all)),
        )
