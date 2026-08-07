"""Handling strategies for missing modalities during multi-modal learning."""

import abc
import logging
from typing import Dict

import numpy as np

from bio_oncograph.research.multimodal.config import MultiModalConfig

logger = logging.getLogger(__name__)


class MissingModalityHandler(abc.ABC):
    """Abstract base class for handling missing modality data."""

    @abc.abstractmethod
    def handle(
        self, features: Dict[str, np.ndarray], config: MultiModalConfig
    ) -> Dict[str, np.ndarray]:
        """Process a feature dictionary, applying a strategy for missing modalities.

        Args:
            features: Existing modality features for a sample.
            config: Multi-modal configuration detailing supported modalities.

        Returns:
            A complete or processed dictionary of modality features.
        """
        pass


class MaskingHandler(MissingModalityHandler):
    """Handler that provides zeroed arrays and optional mask flags for missing data."""

    def handle(
        self, features: Dict[str, np.ndarray], config: MultiModalConfig
    ) -> Dict[str, np.ndarray]:
        """Insert zeroed arrays for any supported modalities not present in features.

        Args:
            features: Existing modality features.
            config: Configuration specifying all supported modalities and dims.

        Returns:
            Dictionary with missing modalities replaced by zero arrays.
        """
        logger.info("Applying MaskingHandler for missing modalities.")
        complete_features = dict(features)
        for mod in config.supported_modalities:
            if mod not in complete_features:
                complete_features[mod] = np.zeros(config.embedding_dim)
                # In practice, an auxiliary mask tensor would track this
        return complete_features


class ImputationHandler(MissingModalityHandler):
    """Handler that imputes missing modalities using basic strategies."""

    def __init__(self, strategy: str = "mean") -> None:
        """Initialize the imputation handler.

        Args:
            strategy: 'mean' or 'zero' imputation.
        """
        self.strategy = strategy

    def handle(
        self, features: Dict[str, np.ndarray], config: MultiModalConfig
    ) -> Dict[str, np.ndarray]:
        """Impute missing data with mean representations or zeros.

        Args:
            features: Existing modality features.
            config: Configuration specifying all supported modalities.

        Returns:
            Dictionary with missing modalities imputed.
        """
        logger.info("Applying ImputationHandler (strategy=%s)", self.strategy)
        complete_features = dict(features)
        
        all_features = [f for f in features.values()]
        mean_feat = np.mean(all_features, axis=0) if all_features else np.zeros(config.embedding_dim)

        for mod in config.supported_modalities:
            if mod not in complete_features:
                if self.strategy == "mean":
                    complete_features[mod] = np.copy(mean_feat)
                else:
                    complete_features[mod] = np.zeros(config.embedding_dim)
        return complete_features


class DropHandler(MissingModalityHandler):
    """Handler that signals dropping a sample entirely if any modality is missing."""

    def handle(
        self, features: Dict[str, np.ndarray], config: MultiModalConfig
    ) -> Dict[str, np.ndarray]:
        """Check for missing modalities and return an empty dictionary if any are absent.

        Args:
            features: Existing modality features.
            config: Configuration specifying required modalities.

        Returns:
            The original features if complete, or an empty dictionary if incomplete.
        """
        logger.info("Applying DropHandler for missing modalities.")
        for mod in config.supported_modalities:
            if mod not in features:
                logger.warning("Modality %s missing. Dropping sample.", mod)
                return {}
        return dict(features)
