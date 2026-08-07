"""Loss outputs."""

import dataclasses
from typing import Dict, Any

@dataclasses.dataclass(frozen=True)
class LossOutput:
    """Output from a loss calculation.
    
    Attributes:
        total_loss: The final computed loss value.
        component_losses: Dictionary of individual loss components.
        weights: Dictionary of weights applied to each component.
        metadata: Additional metadata from the loss calculation.
    """
    total_loss: float
    component_losses: Dict[str, float]
    weights: Dict[str, float]
    metadata: Dict[str, Any]
