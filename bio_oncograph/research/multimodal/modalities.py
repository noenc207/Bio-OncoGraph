"""Modality encoders and registry for multi-modal processing."""

import abc
import logging
from typing import Dict, Type

import numpy as np

logger = logging.getLogger(__name__)


class ModalityEncoder(abc.ABC):
    """Abstract base class for all modality encoders."""

    @abc.abstractmethod
    def encode(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Encode raw or pre-processed data into a modality-specific representation.

        Args:
            data: A dictionary containing data arrays to be encoded.

        Returns:
            A numpy array containing the encoded modality representation.
        """
        pass


class HistopathologyEncoder(ModalityEncoder):
    """Encoder for histopathology image features."""

    def encode(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Mock encoding of histopathology data.

        Args:
            data: Dictionary with histopathology data.

        Returns:
            Mocked numpy array representation.
        """
        logger.info("Encoding histopathology data.")
        image_data = data.get("image", np.zeros((100, 100, 3)))
        return np.mean(image_data, axis=(0, 1))


class ClinicalEncoder(ModalityEncoder):
    """Encoder for clinical and patient metadata features."""

    def encode(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Mock encoding of clinical data.

        Args:
            data: Dictionary with clinical data.

        Returns:
            Mocked numpy array representation.
        """
        logger.info("Encoding clinical data.")
        clinical_features = data.get("clinical", np.zeros((10,)))
        return clinical_features * 0.1


class GenomicsEncoder(ModalityEncoder):
    """Encoder for genomics sequences and mutations."""

    def encode(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Mock encoding of genomics data.

        Args:
            data: Dictionary with genomics data.

        Returns:
            Mocked numpy array representation.
        """
        logger.info("Encoding genomics data.")
        genomics_sequence = data.get("sequence", np.zeros((1000,)))
        return np.sum(genomics_sequence.reshape(-1, 10), axis=1)


class ModalityRegistry:
    """Registry to manage and retrieve modality encoders."""

    def __init__(self) -> None:
        """Initialize an empty modality registry."""
        self._registry: Dict[str, Type[ModalityEncoder]] = {}

    def register(self, name: str, encoder_cls: Type[ModalityEncoder]) -> None:
        """Register a new modality encoder class under a given name.

        Args:
            name: The name of the modality.
            encoder_cls: The encoder class for the modality.
        """
        logger.info("Registering modality encoder for %s", name)
        self._registry[name] = encoder_cls

    def get_encoder(self, name: str) -> ModalityEncoder:
        """Retrieve an instance of the modality encoder by name.

        Args:
            name: The name of the modality.

        Returns:
            An instantiated modality encoder.

        Raises:
            ValueError: If the modality is not registered.
        """
        if name not in self._registry:
            logger.error("Modality %s not found in registry.", name)
            raise ValueError(f"Modality '{name}' is not registered.")
        return self._registry[name]()
