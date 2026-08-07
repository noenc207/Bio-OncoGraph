"""Base class for dataset connectors."""

import abc
from typing import Any, Dict, List, Optional, Tuple

from bio_oncograph.infrastructure.datasets.connectors.metadata import DatasetMetadata


class DatasetConnector(abc.ABC):
    """Abstract base class for dataset connectors.
    
    This class defines the interface that all dataset connectors must implement.
    """

    @abc.abstractmethod
    def load(self, split: Optional[str] = None) -> Any:
        """Loads the dataset.

        Args:
            split: Optional split to load (e.g., 'train', 'test'). Defaults to None.

        Returns:
            The loaded dataset.
        """
        pass

    @abc.abstractmethod
    def validate(self) -> bool:
        """Validates the dataset integrity.

        Returns:
            True if the dataset is valid, False otherwise.
        """
        pass

    @abc.abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Returns the dataset metadata.

        Returns:
            The dataset metadata.
        """
        pass

    @abc.abstractmethod
    def list_samples(self) -> List[str]:
        """Lists all sample identifiers in the dataset.

        Returns:
            A list of sample identifiers.
        """
        pass

    @abc.abstractmethod
    def split(self, ratios: Tuple[float, ...]) -> Dict[str, 'DatasetConnector']:
        """Splits the dataset into multiple subsets.

        Args:
            ratios: A tuple of ratios for each split. Must sum to 1.0.

        Returns:
            A dictionary mapping split names to new DatasetConnector instances.
        """
        pass

    @abc.abstractmethod
    def cache(self) -> None:
        """Caches the dataset locally."""
        pass

    @abc.abstractmethod
    def version(self) -> str:
        """Returns the current version of the dataset.

        Returns:
            The dataset version string.
        """
        pass
