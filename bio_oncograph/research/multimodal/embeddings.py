"""Data structures for storing single and fused modality embeddings."""

import dataclasses
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ModalityEmbedding:
    """Represents a single modality's feature embedding.

    Attributes:
        modality_name: Name of the modality.
        embedding: Numpy array containing the feature representation.
        is_imputed: True if the embedding was generated via imputation.
        metadata: Dictionary containing contextual metadata.
    """
    modality_name: str
    embedding: np.ndarray
    is_imputed: bool
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class FusedEmbedding:
    """Represents a fused embedding derived from multiple modalities.

    Attributes:
        fused_representation: Numpy array containing the fused representation.
        source_modalities: Tuple of modality names that contributed.
        metadata: Dictionary containing contextual metadata.
        version: Version string for tracking embedding models.
    """
    fused_representation: np.ndarray
    source_modalities: Tuple[str, ...]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    version: str = "1.0"


class EmbeddingStore:
    """Manages saving and loading of modality embeddings."""

    def __init__(self) -> None:
        """Initialize the embedding store."""
        self._store: Dict[str, FusedEmbedding] = {}
        logger.info("Initialized EmbeddingStore.")

    def save(self, embedding_id: str, embedding: FusedEmbedding) -> None:
        """Save a fused embedding to the store.

        Args:
            embedding_id: Unique identifier for the embedding.
            embedding: The FusedEmbedding instance to save.
        """
        logger.debug("Saving embedding %s.", embedding_id)
        self._store[embedding_id] = embedding

    def load(self, embedding_id: str) -> Optional[FusedEmbedding]:
        """Load a fused embedding from the store.

        Args:
            embedding_id: Unique identifier for the embedding.

        Returns:
            The requested FusedEmbedding, or None if not found.
        """
        logger.debug("Loading embedding %s.", embedding_id)
        return self._store.get(embedding_id)
