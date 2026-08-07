"""Fusion strategies for combining multi-modal feature representations."""

import abc
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class FusionStrategy(abc.ABC):
    """Abstract base class for multi-modal fusion strategies."""

    @abc.abstractmethod
    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse multiple aligned modality features into a single representation.

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            A fused numpy array representing the combined modalities.
        """
        pass


class EarlyFusion(FusionStrategy):
    """Fuses modalities by concatenating them along the feature dimension."""

    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate features from all modalities.

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            Concatenated numpy array.
        """
        logger.info("Applying EarlyFusion strategy.")
        if not aligned_features:
            return np.array([])
        sorted_mods = sorted(aligned_features.keys())
        features = [aligned_features[mod] for mod in sorted_mods]
        return np.concatenate(features, axis=-1)


class IntermediateFusion(FusionStrategy):
    """Fuses modalities by averaging intermediate representations."""

    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Sum or average aligned features.

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            Mean numpy array across modalities.
        """
        logger.info("Applying IntermediateFusion strategy.")
        if not aligned_features:
            return np.array([])
        features = list(aligned_features.values())
        return np.mean(np.stack(features), axis=0)


class LateFusion(FusionStrategy):
    """Mocks late fusion, typically used for ensembling predictions."""

    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Mock ensembling by a weighted combination.

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            Ensemble-weighted numpy array.
        """
        logger.info("Applying LateFusion strategy.")
        if not aligned_features:
            return np.array([])
        features = list(aligned_features.values())
        # Mock ensembling with equal weights
        weights = np.ones(len(features)) / len(features)
        fused = np.zeros_like(features[0])
        for w, f in zip(weights, features):
            fused += w * f
        return fused


class HierarchicalFusion(FusionStrategy):
    """Mocks a tree-like hierarchical fusion structure."""

    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse features hierarchically (mocked as sequential summation).

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            Hierarchically fused numpy array.
        """
        logger.info("Applying HierarchicalFusion strategy.")
        if not aligned_features:
            return np.array([])
        fused = None
        for mod, feat in aligned_features.items():
            if fused is None:
                fused = feat
            else:
                # Mock a hierarchical gating or combination
                fused = (fused + feat) * 0.5
        return fused if fused is not None else np.array([])


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class AttentionFusion(FusionStrategy):
    """Mocks a self-attention mechanism across modalities."""

    def fuse(self, aligned_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply a mock self-attention across modality features.

        Args:
            aligned_features: Dictionary of aligned feature arrays.

        Returns:
            Attention-fused numpy array.
        """
        logger.info("Applying AttentionFusion strategy.")
        if not aligned_features:
            return np.array([])
        features_stack = np.stack(list(aligned_features.values()))
        # Mock attention weights
        scores = _softmax(np.random.randn(len(aligned_features)))
        attention_weights = scores.reshape(-1, 1)
        return np.sum(features_stack * attention_weights, axis=0)
