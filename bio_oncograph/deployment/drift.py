"""Drift detection module for monitoring model inputs and outputs."""
import abc
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

class DriftDetector(abc.ABC):
    """Abstract base class for drift detectors."""

    @abc.abstractmethod
    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detects drift between reference and current data.

        Args:
            reference_data: The baseline data.
            current_data: The new data to compare against the baseline.

        Returns:
            A dictionary containing drift metrics and a boolean indicating if drift was detected.
        """
        pass


class DataDriftDetector(DriftDetector):
    """Detects drift in general data distributions."""

    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detects data drift using statistical divergence.

        Args:
            reference_data: The baseline data.
            current_data: The new data.

        Returns:
            Dictionary with drift metrics.
        """
        logger.info("Detecting data drift.")
        ref_mean = np.mean(reference_data)
        curr_mean = np.mean(current_data)
        diff = float(np.abs(ref_mean - curr_mean))
        is_drift = diff > 0.1
        return {"drift_detected": is_drift, "mean_difference": diff}


class FeatureDriftDetector(DriftDetector):
    """Detects drift in individual features."""

    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detects feature-level drift.

        Args:
            reference_data: The baseline feature data.
            current_data: The new feature data.

        Returns:
            Dictionary with drift metrics per feature.
        """
        logger.info("Detecting feature drift.")
        diffs = np.abs(np.mean(reference_data, axis=0) - np.mean(current_data, axis=0))
        max_diff = float(np.max(diffs))
        return {"drift_detected": max_diff > 0.1, "max_feature_difference": max_diff}


class PredictionDriftDetector(DriftDetector):
    """Detects drift in model predictions."""

    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detects prediction drift.

        Args:
            reference_data: The baseline prediction data.
            current_data: The new prediction data.

        Returns:
            Dictionary with prediction drift metrics.
        """
        logger.info("Detecting prediction drift.")
        ref_mean = np.mean(reference_data)
        curr_mean = np.mean(current_data)
        diff = float(np.abs(ref_mean - curr_mean))
        return {"drift_detected": diff > 0.05, "prediction_mean_difference": diff}


class EmbeddingDriftDetector(DriftDetector):
    """Detects drift in latent representations or embeddings."""

    def detect(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Detects embedding drift using distance metrics.

        Args:
            reference_data: The baseline embeddings.
            current_data: The new embeddings.

        Returns:
            Dictionary with embedding drift metrics.
        """
        logger.info("Detecting embedding drift.")
        ref_norm = np.linalg.norm(reference_data, axis=1).mean()
        curr_norm = np.linalg.norm(current_data, axis=1).mean()
        diff = float(np.abs(ref_norm - curr_norm))
        return {"drift_detected": diff > 0.5, "norm_difference": diff}
