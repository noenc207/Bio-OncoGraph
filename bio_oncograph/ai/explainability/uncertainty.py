"""Uncertainty quantification methods."""

import numpy as np


def compute_aleatoric_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """Computes aleatoric uncertainty (entropy) from prediction probabilities.

    Args:
        predictions: Array of shape (batch_size, num_classes) containing
            predicted probabilities.

    Returns:
        Array of shape (batch_size,) containing entropy values.
    """
    epsilon = 1e-12
    safe_preds = np.clip(predictions, epsilon, 1.0)
    entropy = -np.sum(safe_preds * np.log(safe_preds), axis=-1)
    return entropy


def compute_epistemic_uncertainty(mc_predictions: np.ndarray) -> np.ndarray:
    """Computes epistemic uncertainty from Monte Carlo predictions.

    Args:
        mc_predictions: Array of shape (num_samples, batch_size, num_classes)
            containing predicted probabilities from multiple forward passes
            (e.g., using Monte Carlo dropout).

    Returns:
        Array of shape (batch_size,) containing variance-based uncertainty values.
    """
    mean_preds = np.mean(mc_predictions, axis=0)
    # Variance across the MC samples, averaged over classes
    variance = np.var(mc_predictions, axis=0)
    epistemic_uncertainty = np.mean(variance, axis=-1)
    return epistemic_uncertainty
