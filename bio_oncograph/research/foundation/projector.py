"""Projection heads for self-supervised learning."""

import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Projector(abc.ABC):
    """Abstract base class for projection heads."""

    @abc.abstractmethod
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Project embeddings to a new space."""
        pass

class MLPProjector(Projector):
    """Two-layer MLP projection head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._w1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
        self._b1 = np.zeros(hidden_dim)
        self._w2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
        self._b2 = np.zeros(output_dim)

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        hidden = embeddings @ self._w1 + self._b1
        hidden = np.maximum(0, hidden)  # ReLU
        return hidden @ self._w2 + self._b2

class LinearProjector(Projector):
    """Single linear layer projection head."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._w = rng.normal(0, 0.1, (input_dim, output_dim))
        self._b = np.zeros(output_dim)

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings @ self._w + self._b

class IdentityProjector(Projector):
    """Identity projection head."""

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings
