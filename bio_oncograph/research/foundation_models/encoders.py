"""Encoders using foundation model adapters."""

import abc
import logging

import numpy as np

from bio_oncograph.research.foundation_models.adapters import FoundationAdapter


logger = logging.getLogger(__name__)


class BaseEncoder(abc.ABC):
    """Abstract base class for encoders."""

    def __init__(self, adapter: FoundationAdapter) -> None:
        """Initializes the encoder with an adapter.

        Args:
            adapter: The foundation adapter to use.
        """
        self.adapter = adapter
        self.adapter.load_weights()

    @abc.abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes data into representations.

        Args:
            data: Input numpy array.

        Returns:
            Encoded representations as a numpy array.
        """
        pass


class ImageEncoder(BaseEncoder):
    """Encoder for image data."""

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes image data."""
        logger.info("Encoding image data.")
        return self.adapter.forward({'image': data})


class TextEncoder(BaseEncoder):
    """Encoder for text data."""

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes text data."""
        logger.info("Encoding text data.")
        return self.adapter.forward({'text': data})


class GraphEncoder(BaseEncoder):
    """Encoder for graph data."""

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encodes graph data."""
        logger.info("Encoding graph data.")
        return self.adapter.forward({'graph': data})
