"""Validation functions for explainability inputs."""

from typing import Any, Dict, Optional
import numpy as np


def validate_explainability_inputs(
    model_output: Dict[str, Any],
    graph: Optional[Any] = None
) -> None:
    """Validates the input to the explainability engine.

    Args:
        model_output: Dictionary containing model outputs (must contain 'logits' and 'predictions').
        graph: Optional graph representation used in the model.

    Raises:
        ValueError: If inputs are invalid, missing keys, or contain NaN/Inf.
    """
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary.")

    required_keys = ['logits', 'predictions']
    for key in required_keys:
        if key not in model_output:
            raise ValueError(f"model_output is missing required key: {key}")

    for key, val in model_output.items():
        if isinstance(val, np.ndarray):
            if not np.issubdtype(val.dtype, np.number):
                continue
            if np.isnan(val).any():
                raise ValueError(f"NaN found in model_output key: {key}")
            if np.isinf(val).any():
                raise ValueError(f"Infinity found in model_output key: {key}")
