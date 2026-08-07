"""Classification loss functions."""

import numpy as np
from bio_oncograph.ai.losses.base import LossFunction


class CrossEntropyLoss(LossFunction):
    """Cross entropy loss for multi-class classification."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        """Initialize the cross entropy loss.

        Args:
            epsilon: Small value to prevent log(0).
        """
        super().__init__()
        self.epsilon = epsilon

    def compute(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cross entropy loss.

        Args:
            outputs: Predicted probabilities.
            targets: True labels (one-hot encoded).

        Returns:
            Scalar loss value.
        """
        outputs = np.clip(outputs, self.epsilon, 1. - self.epsilon)
        return -np.mean(np.sum(targets * np.log(outputs), axis=-1))


class BinaryCrossEntropyLoss(LossFunction):
    """Binary cross entropy loss."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        """Initialize the binary cross entropy loss.

        Args:
            epsilon: Small value to prevent log(0).
        """
        super().__init__()
        self.epsilon = epsilon

    def compute(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute binary cross entropy loss.

        Args:
            outputs: Predicted probabilities.
            targets: True labels (0 or 1).

        Returns:
            Scalar loss value.
        """
        outputs = np.clip(outputs, self.epsilon, 1. - self.epsilon)
        loss = -(targets * np.log(outputs) + (1.0 - targets) * np.log(1.0 - outputs))
        return np.mean(loss)
