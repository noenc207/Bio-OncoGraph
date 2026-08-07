"""Survival analysis loss functions."""

import numpy as np
from bio_oncograph.ai.losses.base import LossFunction


class CoxLoss(LossFunction):
    """Cox proportional hazards loss function."""

    def __init__(self, epsilon: float = 1e-15) -> None:
        """Initialize the Cox loss.

        Args:
            epsilon: Small value for numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon

    def compute(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute Cox loss (negative log partial likelihood).

        Args:
            outputs: Predicted risk scores (shape: [N]).
            targets: Survival targets containing time (first col) and event (second col).
                     Shape: [N, 2].

        Returns:
            Scalar loss value.
        """
        times = targets[:, 0]
        events = targets[:, 1]
        
        # Sort by time descending
        sort_idx = np.argsort(-times)
        sorted_times = times[sort_idx]
        sorted_events = events[sort_idx]
        sorted_outputs = outputs[sort_idx]

        loss = 0.0
        n_samples = len(sorted_times)
        
        # Risk set sum tracker
        max_output = np.max(sorted_outputs) if n_samples > 0 else 0.0
        exp_outputs = np.exp(sorted_outputs - max_output)

        cum_risk = np.cumsum(exp_outputs)
        
        # Basic Breslow approximation loop
        valid_events = 0
        for i in range(n_samples):
            if sorted_events[i] == 1:
                # Risk set at time t is all individuals with time >= t.
                # Since times are descending, for index i, individuals with time >= sorted_times[i] are indices 0 to i.
                risk_sum = cum_risk[i]
                loss += (sorted_outputs[i] - max_output) - np.log(risk_sum + self.epsilon)
                valid_events += 1

        if valid_events > 0:
            return np.array(-loss / valid_events)
        return np.array(0.0)
