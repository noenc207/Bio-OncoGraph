"""
Transform pipeline for Bio-OncoGraph.

Provides composable, deterministic data transforms:
  - ``Transform`` — abstract base class.
  - ``Identity`` — no-op pass-through.
  - ``LambdaTransform`` — wraps an arbitrary callable.
  - ``Compose`` — sequential composition of transforms.

All transforms are stateless and operate on ``DataSample`` instances.
**No** ``torch``, ``numpy``, or ``pandas`` imports.

Design references:
  - Software Architecture § 8 (Transform Pipeline)
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Sequence

from bio_oncograph.core.data import DataSample

logger = logging.getLogger("BioOncoGraph")


class Transform(ABC):
    """
    Abstract base class for data transforms.

    Subclasses must implement ``apply`` to produce a new
    ``DataSample`` from an existing one.  Transforms are also
    callable via ``__call__`` for convenience.
    """

    @abstractmethod
    def apply(self, sample: DataSample) -> DataSample:
        """
        Apply the transform to a data sample.

        Args:
            sample: The input data sample.

        Returns:
            A (possibly new) ``DataSample``.
        """
        ...

    def __call__(self, sample: DataSample) -> DataSample:
        """Alias for ``apply``."""
        return self.apply(sample)


class Identity(Transform):
    """
    No-op transform that returns the sample unchanged.

    Useful as a default placeholder in transform pipelines.
    """

    def apply(self, sample: DataSample) -> DataSample:
        """
        Return the sample unmodified.

        Args:
            sample: The input data sample.

        Returns:
            The same ``DataSample`` instance.
        """
        return sample

    def __repr__(self) -> str:
        return "Identity()"


class LambdaTransform(Transform):
    """
    Transform that wraps an arbitrary callable.

    The callable must accept a ``DataSample`` and return a
    ``DataSample``.

    Args:
        fn: The transformation function.
        name: Optional human-readable name for logging.
    """

    def __init__(
        self,
        fn: Callable[[DataSample], DataSample],
        name: str = "lambda",
    ) -> None:
        """Initialise with a callable."""
        if not callable(fn):
            raise TypeError(
                f"fn must be callable, got {type(fn).__name__}."
            )
        self._fn = fn
        self._name = name

    def apply(self, sample: DataSample) -> DataSample:
        """
        Apply the wrapped callable to the sample.

        Args:
            sample: The input data sample.

        Returns:
            The transformed ``DataSample``.
        """
        return self._fn(sample)

    def __repr__(self) -> str:
        return f"LambdaTransform(name={self._name!r})"


class Compose(Transform):
    """
    Sequential composition of transforms.

    Transforms are applied left-to-right in the order they were
    provided.

    Args:
        transforms: Ordered sequence of transforms to compose.
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """Initialise with an ordered sequence of transforms."""
        if not transforms:
            raise ValueError("Compose requires at least one transform.")
        for i, t in enumerate(transforms):
            if not isinstance(t, Transform):
                raise TypeError(
                    f"Element {i} is not a Transform, "
                    f"got {type(t).__name__}."
                )
        self._transforms: List[Transform] = list(transforms)

    def apply(self, sample: DataSample) -> DataSample:
        """
        Apply all transforms sequentially.

        Args:
            sample: The input data sample.

        Returns:
            The result after all transforms have been applied.
        """
        for transform in self._transforms:
            sample = transform.apply(sample)
        return sample

    @property
    def transforms(self) -> List[Transform]:
        """Return a copy of the transform list."""
        return list(self._transforms)

    def __len__(self) -> int:
        """Return the number of composed transforms."""
        return len(self._transforms)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self._transforms)
        return f"Compose([{inner}])"
