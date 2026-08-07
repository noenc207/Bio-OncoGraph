"""Regularization loss functions."""

import numpy as np
from bio_oncograph.ai.losses.base import LossFunction


class L1Loss(LossFunction):
    """L1 regularization loss."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize L1 loss.

        Args:
            weight: Regularization coefficient.
        """
        super().__init__()
        self.weight = weight

    def compute(self, outputs: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        """Compute L1 loss on the parameters.

        Args:
            outputs: Model parameters.
            targets: Ignored for regularization.

        Returns:
            Scalar loss value.
        """
        return np.array(self.weight * np.sum(np.abs(outputs)))


class L2Loss(LossFunction):
    """L2 regularization loss."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize L2 loss.

        Args:
            weight: Regularization coefficient.
        """
        super().__init__()
        self.weight = weight

    def compute(self, outputs: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        """Compute L2 loss on the parameters.

        Args:
            outputs: Model parameters.
            targets: Ignored for regularization.

        Returns:
            Scalar loss value.
        """
        return np.array(self.weight * np.sum(np.square(outputs)))


class WeightDecay(LossFunction):
    """Weight decay (equivalent to L2 regularization scaled by 0.5)."""

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize weight decay.

        Args:
            weight: Regularization coefficient.
        """
        super().__init__()
        self.weight = weight

    def compute(self, outputs: np.ndarray, targets: np.ndarray = None) -> np.ndarray:
        """Compute weight decay on the parameters.

        Args:
            outputs: Model parameters.
            targets: Ignored for regularization.

        Returns:
            Scalar loss value.
        """
        return np.array(0.5 * self.weight * np.sum(np.square(outputs)))
