"""Base loss function."""

import abc
from typing import Any, Dict

class LossFunction(abc.ABC):
    """Abstract base class for all loss functions."""
    
    @abc.abstractmethod
    def compute(self, outputs: Any, targets: Any) -> float:
        """Compute the loss.
        
        Args:
            outputs: Network outputs.
            targets: Target values.
            
        Returns:
            Computed loss value.
        """
        pass
        
    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate the loss configuration.
        
        Returns:
            True if valid, False otherwise.
        """
        pass
        
    @abc.abstractmethod
    def report(self) -> Dict[str, Any]:
        """Generate a report of the loss state.
        
        Returns:
            Dictionary containing reporting metrics.
        """
        pass
