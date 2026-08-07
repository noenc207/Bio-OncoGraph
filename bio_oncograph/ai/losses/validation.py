"""Loss validation utilities."""

import numpy as np

def validate_loss_inputs(outputs: np.ndarray, targets: np.ndarray) -> None:
    """Validate inputs for loss calculation.
    
    Args:
        outputs: Network outputs.
        targets: Target values.
        
    Raises:
        ValueError: If shapes mismatch, or if inputs contain NaN or Inf.
    """
    if outputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: outputs shape {outputs.shape} != targets shape {targets.shape}")
        
    if np.isnan(outputs).any() or np.isnan(targets).any():
        raise ValueError("NaN values found in loss inputs")
        
    if np.isinf(outputs).any() or np.isinf(targets).any():
        raise ValueError("Inf values found in loss inputs")
