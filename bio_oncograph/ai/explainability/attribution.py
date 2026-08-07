"""Attribution methods for explainability."""

from typing import Protocol

import numpy as np


class AttributionMethod(Protocol):
    """Protocol for attribution methods."""

    def attribute(self, inputs: np.ndarray) -> np.ndarray:
        """Computes attribution scores for the given inputs.

        Args:
            inputs: The input features.

        Returns:
            The attribution scores.
        """
        ...


class GradientAttribution:
    """Mock gradient-based attribution method."""

    def __init__(self) -> None:
        """Initializes the GradientAttribution."""
        pass

    def attribute(self, inputs: np.ndarray) -> np.ndarray:
        """Computes mock gradient attribution scores.

        Args:
            inputs: The input features.

        Returns:
            Attribution scores proportional to inputs, normalized.
        """
        scores = np.abs(inputs)
        max_val = np.max(scores)
        if max_val > 0:
            return scores / max_val
        return scores


class AttentionAttribution:
    """Mock attention-based attribution method."""

    def __init__(self) -> None:
        """Initializes the AttentionAttribution."""
        pass

    def attribute(self, inputs: np.ndarray) -> np.ndarray:
        """Computes mock attention attribution scores.

        Args:
            inputs: The input features.

        Returns:
            Attribution scores based on softmax of inputs.
        """
        # Subtract max for numerical stability
        shifted_inputs = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted_inputs)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
