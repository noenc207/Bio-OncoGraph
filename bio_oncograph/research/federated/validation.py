import logging
from dataclasses import dataclass
from typing import Any, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationResult:
    """Immutable data class for validation results."""
    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

class FederatedValidator:
    """Validator for federated learning components and processes."""
    
    def __init__(self) -> None:
        """Initializes the FederatedValidator."""
        logger.info("Initialized FederatedValidator.")
        
    def validate_client(self, client: Any) -> ValidationResult:
        """Validates a federated learning client.
        
        Args:
            client: The client object to validate.
            
        Returns:
            ValidationResult indicating client validity.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        if not hasattr(client, 'client_id'):
            errors.append("Client missing 'client_id' attribute.")
            
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, tuple(errors), tuple(warnings))
        
    def validate_aggregation(self, updates: List[np.ndarray]) -> ValidationResult:
        """Validates a collection of client updates before aggregation.
        
        Args:
            updates: List of client updates (numpy arrays).
            
        Returns:
            ValidationResult indicating aggregation readiness.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        if not updates:
            errors.append("Empty updates list.")
            return ValidationResult(False, tuple(errors), tuple(warnings))
            
        expected_shape = updates[0].shape
        for i, update in enumerate(updates):
            if update.shape != expected_shape:
                errors.append(f"Update {i} shape mismatch: {update.shape} != {expected_shape}.")
            if np.isnan(update).any():
                errors.append(f"Update {i} contains NaNs.")
                
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, tuple(errors), tuple(warnings))
        
    def validate_round(self, round_info: Dict[str, Any]) -> ValidationResult:
        """Validates round information.
        
        Args:
            round_info: Dictionary containing round metrics and state.
            
        Returns:
            ValidationResult indicating round validity.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        if 'round_number' not in round_info:
            errors.append("Round info missing 'round_number'.")
            
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, tuple(errors), tuple(warnings))
