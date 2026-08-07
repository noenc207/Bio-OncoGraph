"""
Abstract Dataset interface for Bio-OncoGraph.

Defines ``IDataset`` — the contract that all dataset implementations
must fulfil.  The interface lives in the Domain layer and has **no**
``torch``, ``numpy``, or ``pandas`` dependencies.

Design references:
  - Software Architecture § 6 (Data Layer)
  - Software Architecture § 3 (Clean Architecture — Domain Interfaces)
"""

from abc import ABC, abstractmethod

from bio_oncograph.core.data import DataSample, DatasetMetadata


class IDataset(ABC):
    """
    Abstract dataset interface.

    All dataset implementations — regardless of backend or storage
    format — must satisfy this contract so that the ``Trainer`` and
    ``PipelineBuilder`` can consume them interchangeably.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Non-negative integer count.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> DataSample:
        """
        Retrieve a single sample by index.

        Args:
            index: Zero-based sample index.

        Returns:
            The ``DataSample`` at the given index.

        Raises:
            IndexError: If *index* is out of bounds.
        """
        ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """
        Return the dataset metadata descriptor.

        Returns:
            An immutable ``DatasetMetadata`` instance.
        """
        ...

    @abstractmethod
    def validate(self) -> None:
        """
        Validate dataset integrity.

        Checks for missing labels, duplicate IDs, metadata
        consistency, etc.

        Raises:
            ValueError: If any validation check fails.
        """
        ...
