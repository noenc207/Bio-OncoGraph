"""CalTopoSurv loss function."""

import numpy as np
from bio_oncograph.ai.losses.base import LossFunction


class CalTopoSurvLoss(LossFunction):
    """Topological survival calibration loss mock/approximation."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        """Initialize the CalTopoSurv loss.

        Args:
            epsilon: Small value for numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon

    def compute(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute continuous calibration loss using time and event features.

        Args:
            outputs: Predicted risks or survival times.
            targets: True survival data containing time (col 0) and event (col 1).

        Returns:
            Scalar calibration loss value.
        """
        times = targets[:, 0]
        events = targets[:, 1]
        
        # Normalize outputs and times for stable computation
        mean_out = np.mean(outputs)
        std_out = np.std(outputs) + self.epsilon
        norm_outputs = (outputs - mean_out) / std_out
        
        mean_time = np.mean(times)
        std_time = np.std(times) + self.epsilon
        norm_times = (times - mean_time) / std_time
        
        # A mock topological calibration comparing orderings and distances
        # We penalize disagreement between predicted risk and actual time
        diff = norm_outputs + norm_times  # Higher risk should mean lower time, so out ~ -time
        
        loss = np.mean(events * np.square(diff))
        return np.array(loss)
