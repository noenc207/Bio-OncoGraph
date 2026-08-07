"""Validation module for Bio-OncoGraph pipeline outputs."""

from typing import Any, Dict
import numpy as np


class ScientificValidator:
    """Validates the outputs of the scientific pipeline."""

    def __init__(self) -> None:
        """Initializes the ScientificValidator."""
        pass

    def validate_pipeline_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Verifies pipeline outputs for NaNs, Infs, and bounds.

        Args:
            outputs: A dictionary containing the pipeline outputs.

        Returns:
            True if all outputs are valid, False otherwise.
        """
        if not outputs:
            return False

        for _, value in outputs.items():
            if isinstance(value, np.ndarray):
                if not self._validate_array(value):
                    return False
            elif isinstance(value, float):
                if not self._validate_float(value):
                    return False
            elif isinstance(value, dict):
                if not self.validate_pipeline_outputs(value):
                    return False
        
        # Specific bounds check for trust score if present
        if "trust_score" in outputs:
            trust_score = outputs["trust_score"]
            if not isinstance(trust_score, (float, np.floating)):
                return False
            if trust_score < 0.0 or trust_score > 1.0:
                return False

        return True

    def _validate_array(self, arr: np.ndarray) -> bool:
        """Checks if a numpy array contains NaNs or Infs."""
        if not np.issubdtype(arr.dtype, np.number):
            return True 
        if np.any(np.isnan(arr)):
            return False
        if np.any(np.isinf(arr)):
            return False
        return True

    def _validate_float(self, val: float) -> bool:
        """Checks if a float is NaN or Inf."""
        if np.isnan(val) or np.isinf(val):
            return False
        return True
