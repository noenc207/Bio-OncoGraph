"""Model calibration metrics."""

import numpy as np


def compute_brier_score(
    predictions: np.ndarray, targets: np.ndarray
) -> float:
    """Computes the Brier Score for multi-class classification.

    Args:
        predictions: Array of shape (batch_size, num_classes) containing
            predicted probabilities.
        targets: Array of shape (batch_size,) containing true class indices
            or one-hot encoded targets (batch_size, num_classes).

    Returns:
        The mean Brier Score.
    """
    if targets.ndim == 1:
        num_classes = predictions.shape[1]
        one_hot_targets = np.eye(num_classes)[targets]
    else:
        one_hot_targets = targets

    squared_diff = (predictions - one_hot_targets) ** 2
    brier_score = np.mean(np.sum(squared_diff, axis=1))
    return float(brier_score)


def compute_expected_calibration_error(
    predictions: np.ndarray, targets: np.ndarray, num_bins: int = 10
) -> float:
    """Computes the Expected Calibration Error (ECE).

    Args:
        predictions: Array of shape (batch_size, num_classes) containing
            predicted probabilities.
        targets: Array of shape (batch_size,) containing true class indices.
        num_bins: Number of bins for computing ECE.

    Returns:
        The Expected Calibration Error.
    """
    confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=1)

    accuracies = predicted_classes == targets

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        # Include exactly 0.0 in the first bin
        if i == 0:
            in_bin = in_bin | (confidences == 0.0)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
