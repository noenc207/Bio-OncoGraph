"""Feature alignment strategies for standardizing multi-modal representations."""

import abc
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class FeatureAligner(abc.ABC):
    """Abstract base class for aligning features from different modalities."""

    @abc.abstractmethod
    def align(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align features from different modalities into a common space or format.

        Args:
            features: Dictionary mapping modality names to feature arrays.

        Returns:
            Dictionary mapping modality names to aligned feature arrays.
        """
        pass


class LatentSpaceAligner(FeatureAligner):
    """Aligns features by projecting them into a common latent dimension."""

    def __init__(self, target_dim: int = 256) -> None:
        """Initialize the latent space aligner.

        Args:
            target_dim: The target dimension for the latent space projection.
        """
        self.target_dim = target_dim

    def align(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Mock projection to the same target dimension using padding/truncation.

        Args:
            features: Dictionary mapping modality names to feature arrays.

        Returns:
            Dictionary mapping modality names to aligned feature arrays.
        """
        logger.info("Aligning features using LatentSpaceAligner.")
        aligned_features = {}
        for mod, feat in features.items():
            current_dim = feat.shape[-1] if feat.ndim > 0 else 1
            feat_flat = np.atleast_1d(feat).flatten()
            
            if current_dim < self.target_dim:
                pad_width = self.target_dim - current_dim
                aligned_feat = np.pad(feat_flat, (0, pad_width), mode='constant')
            else:
                aligned_feat = feat_flat[:self.target_dim]
            aligned_features[mod] = aligned_feat
        return aligned_features


class TokenAligner(FeatureAligner):
    """Aligns features formatted as token sequences."""

    def __init__(self, target_length: int = 128, token_dim: int = 64) -> None:
        """Initialize the token aligner.

        Args:
            target_length: Desired sequence length.
            token_dim: Dimension of each token.
        """
        self.target_length = target_length
        self.token_dim = token_dim

    def align(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Mock token sequence padding and alignment.

        Args:
            features: Dictionary mapping modality names to feature sequences.

        Returns:
            Dictionary mapping modality names to aligned token sequences.
        """
        logger.info("Aligning features using TokenAligner.")
        aligned_features = {}
        for mod, feat in features.items():
            aligned_feat = np.zeros((self.target_length, self.token_dim))
            num_tokens = min(feat.shape[0] if feat.ndim > 0 else 1, self.target_length)
            
            if feat.ndim > 1:
                feat_dim = min(feat.shape[1], self.token_dim)
                aligned_feat[:num_tokens, :feat_dim] = feat[:num_tokens, :feat_dim]
            aligned_features[mod] = aligned_feat
        return aligned_features


class EmbeddingAligner(FeatureAligner):
    """Aligns features by performing scaling and normalization."""

    def align(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Mock normalization of embeddings to standard ranges.

        Args:
            features: Dictionary mapping modality names to feature arrays.

        Returns:
            Dictionary mapping modality names to normalized feature arrays.
        """
        logger.info("Aligning features using EmbeddingAligner.")
        aligned_features = {}
        for mod, feat in features.items():
            norm = np.linalg.norm(feat)
            if norm > 0:
                aligned_feat = feat / norm
            else:
                aligned_feat = feat
            aligned_features[mod] = aligned_feat
        return aligned_features
