"""Validation utilities for foundation models."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

class FoundationValidator:
    """Validator for foundation model inputs and outputs."""

    def validate_graph(self, node_features: np.ndarray, edge_index: np.ndarray) -> ValidationResult:
        """Check shapes, NaN, Inf, and edge bounds.

        Args:
            node_features: Node features array.
            edge_index: Edge index array.

        Returns:
            ValidationResult instance.
        """
        errors = []
        warnings = []
        
        if not isinstance(node_features, np.ndarray) or not isinstance(edge_index, np.ndarray):
            errors.append("Inputs must be numpy arrays.")
            return ValidationResult(False, tuple(errors), tuple(warnings))
            
        if len(node_features.shape) != 2:
            errors.append(f"node_features should be 2D, got shape {node_features.shape}")
            
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            errors.append(f"edge_index should be 2xN, got shape {edge_index.shape}")
            
        if np.isnan(node_features).any() or np.isinf(node_features).any():
            errors.append("node_features contains NaN or Inf.")
            
        if edge_index.size > 0:
            num_nodes = node_features.shape[0]
            if edge_index.max() >= num_nodes or edge_index.min() < 0:
                errors.append("edge_index contains out-of-bounds indices.")
                
        return ValidationResult(len(errors) == 0, tuple(errors), tuple(warnings))

    def validate_embedding(self, embedding: np.ndarray) -> ValidationResult:
        """Check shape, NaN, Inf, and zero-vectors.

        Args:
            embedding: Embedding array.

        Returns:
            ValidationResult instance.
        """
        errors = []
        warnings = []
        
        if len(embedding.shape) != 2:
            errors.append("Embedding must be a 2D array.")
            
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            errors.append("Embedding contains NaN or Inf.")
            
        # Check for all-zero vectors
        zero_rows = np.all(embedding == 0, axis=1)
        if np.any(zero_rows):
            warnings.append(f"Found {np.sum(zero_rows)} all-zero embedding vectors.")
            
        return ValidationResult(len(errors) == 0, tuple(errors), tuple(warnings))

    def validate_augmentation(self, original: np.ndarray, augmented: np.ndarray) -> ValidationResult:
        """Check shape preservation and no all-zero output.

        Args:
            original: Original features.
            augmented: Augmented features.

        Returns:
            ValidationResult instance.
        """
        errors = []
        warnings = []
        
        if original.shape != augmented.shape:
            errors.append(f"Shape mismatch: original {original.shape} vs augmented {augmented.shape}")
            
        if np.all(augmented == 0):
            errors.append("Augmented output is all zeros.")
            
        if np.array_equal(original, augmented):
            warnings.append("Augmented output is identical to original.")
            
        return ValidationResult(len(errors) == 0, tuple(errors), tuple(warnings))

    def validate_config(self, config: Any) -> ValidationResult:
        """Check parameter ranges.

        Args:
            config: Configuration object.

        Returns:
            ValidationResult instance.
        """
        errors = []
        warnings = []
        
        if hasattr(config, 'learning_rate'):
            if config.learning_rate <= 0 or config.learning_rate >= 1:
                errors.append("Learning rate must be between 0 and 1 exclusive.")
                
        if hasattr(config, 'epochs'):
            if config.epochs <= 0:
                errors.append("Epochs must be strictly positive.")
                
        return ValidationResult(len(errors) == 0, tuple(errors), tuple(warnings))
