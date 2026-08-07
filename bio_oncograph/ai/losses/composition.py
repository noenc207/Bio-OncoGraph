"""Composite loss function."""

import numpy as np
from typing import Dict, Any

from bio_oncograph.ai.losses.base import LossFunction
from bio_oncograph.ai.losses.outputs import LossOutput
from bio_oncograph.ai.losses.config import LossConfig

class CompositeLoss(LossFunction):
    """Composite loss function that combines multiple losses."""
    
    def __init__(self, losses: Dict[str, LossFunction], weights: Dict[str, float], config: LossConfig):
        """Initialize the composite loss.
        
        Args:
            losses: Dictionary mapping names to LossFunction instances.
            weights: Dictionary mapping names to float weights.
            config: Loss configuration.
        """
        self.losses = losses
        self.weights = weights
        self.config = config
        
    def compute(self, outputs: Any, targets: Any) -> LossOutput:
        """Compute the composite loss.
        
        Args:
            outputs: Network outputs.
            targets: Target values.
            
        Returns:
            LossOutput containing the total loss and components.
        """
        total = 0.0
        components = {}
        
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            try:
                loss_val = loss_fn.compute(outputs, targets)
                # Catch NaNs/Infs and apply epsilon
                if np.isnan(loss_val) or np.isinf(loss_val):
                    loss_val = self.config.epsilon
            except Exception:
                # Fallback to epsilon on error
                loss_val = self.config.epsilon
                
            components[name] = float(loss_val)
            total += weight * float(loss_val)
            
        return LossOutput(
            total_loss=float(total),
            component_losses=components,
            weights=self.weights.copy(),
            metadata={"reduction": self.config.reduction}
        )
        
    def validate(self) -> bool:
        """Validate all component losses.
        
        Returns:
            True if all component losses are valid.
        """
        return all(loss_fn.validate() for loss_fn in self.losses.values())
        
    def report(self) -> Dict[str, Any]:
        """Generate a report of the composite loss.
        
        Returns:
            Dictionary containing the report.
        """
        return {
            "is_valid": self.validate(),
            "components": {name: loss_fn.report() for name, loss_fn in self.losses.items()}
        }
