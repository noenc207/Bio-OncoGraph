"""Inference engine for foundation models."""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Engine handling batching, deterministic execution, and normalization."""

    def __init__(self, seed: int = 42) -> None:
        """Initializes the InferenceEngine.

        Args:
            seed: Random seed for deterministic execution.
        """
        logger.debug("Initializing InferenceEngine with seed %d", seed)
        self.seed = seed
        np.random.seed(self.seed)

    def process_batch(self, inputs: List[Any], batch_size: int = 32) -> List[List[Any]]:
        """Splits inputs into batches.

        Args:
            inputs: List of inputs to process.
            batch_size: Maximum size of each batch.

        Returns:
            List of batches.
        """
        logger.debug("Processing batches with size %d", batch_size)
        return [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]

    def run_deterministic(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Runs a function with deterministic settings.

        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function.
        """
        logger.debug("Running deterministic execution.")
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            result = func(*args, **kwargs)
        finally:
            np.random.set_state(state)
        return result

    def normalize_output(self, outputs: np.ndarray) -> np.ndarray:
        """Normalizes the output array using L2 norm.

        Args:
            outputs: The raw output array.

        Returns:
            The L2-normalized array.
        """
        logger.debug("Normalizing outputs.")
        norm = np.linalg.norm(outputs, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return outputs / norm
