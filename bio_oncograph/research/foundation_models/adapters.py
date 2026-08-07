"""Adapters for foundation models."""

import abc
import logging
from typing import Dict, Type

import numpy as np


logger = logging.getLogger(__name__)


class FoundationAdapter(abc.ABC):
    """Abstract base class for foundation model adapters."""

    @abc.abstractmethod
    def load_weights(self) -> None:
        """Loads model weights."""
        pass

    @abc.abstractmethod
    def forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Performs a forward pass.

        Args:
            inputs: Dictionary of input numpy arrays.

        Returns:
            Output numpy array.
        """
        pass


class PathologyAdapter(FoundationAdapter):
    """Adapter for pathology foundation models."""

    def load_weights(self) -> None:
        """Loads pathology model weights."""
        logger.info("Loading weights for PathologyAdapter.")

    def forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass for pathology data."""
        logger.info("Forward pass in PathologyAdapter.")
        return np.zeros((1, 768))


class VisionTransformerAdapter(FoundationAdapter):
    """Adapter for vision transformer models."""

    def load_weights(self) -> None:
        """Loads vision transformer weights."""
        logger.info("Loading weights for VisionTransformerAdapter.")

    def forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass for vision data."""
        logger.info("Forward pass in VisionTransformerAdapter.")
        return np.zeros((1, 768))


class GraphFoundationAdapter(FoundationAdapter):
    """Adapter for graph foundation models."""

    def load_weights(self) -> None:
        """Loads graph model weights."""
        logger.info("Loading weights for GraphFoundationAdapter.")

    def forward(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass for graph data."""
        logger.info("Forward pass in GraphFoundationAdapter.")
        return np.zeros((1, 768))


class AdapterRegistry:
    """Registry for managing adapters."""

    def __init__(self) -> None:
        """Initializes the registry."""
        self._registry: Dict[str, Type[FoundationAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[FoundationAdapter]) -> None:
        """Registers an adapter class."""
        self._registry[name] = adapter_cls
        logger.info(f"Registered adapter: {name}")

    def get(self, name: str) -> Type[FoundationAdapter]:
        """Gets an adapter class by name."""
        if name not in self._registry:
            raise KeyError(f"Adapter '{name}' not found.")
        return self._registry[name]
