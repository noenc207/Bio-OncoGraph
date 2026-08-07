"""Validation module for foundation models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation check.

    Attributes:
        is_valid: Whether the validation passed.
        errors: List of error messages.
    """
    is_valid: bool
    errors: tuple[str, ...]


class FoundationValidator:
    """Validator for foundation models components."""

    def __init__(self) -> None:
        """Initializes the FoundationValidator."""
        logger.debug("Initializing FoundationValidator.")

    def validate_embedding_dimension(self, embedding: np.ndarray, expected_dim: int) -> ValidationResult:
        """Validates the dimensionality of an embedding array.

        Args:
            embedding: The embedding numpy array.
            expected_dim: The expected dimensionality.

        Returns:
            The validation result.
        """
        logger.debug("Validating embedding dimension.")
        errors = []
        if embedding.shape[-1] != expected_dim:
            errors.append(f"Expected embedding dimension {expected_dim}, got {embedding.shape[-1]}.")
        if np.isnan(embedding).any():
            errors.append("Embedding contains NaNs.")
        if np.isinf(embedding).any():
            errors.append("Embedding contains Infs.")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=tuple(errors))

    def validate_prompt(self, prompt: str, registry: Dict[str, Any]) -> ValidationResult:
        """Validates a prompt against the registry.

        Args:
            prompt: The text prompt.
            registry: The registry of valid keys/models.

        Returns:
            The validation result.
        """
        logger.debug("Validating prompt.")
        errors = []
        if not prompt.strip():
            errors.append("Prompt is empty or only whitespace.")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=tuple(errors))

    def validate_adapter(self, adapter: Any) -> ValidationResult:
        """Validates an adapter interface.

        Args:
            adapter: The adapter object.

        Returns:
            The validation result.
        """
        logger.debug("Validating adapter.")
        errors = []
        if not hasattr(adapter, "forward") and not hasattr(adapter, "encode"):
            errors.append("Adapter is missing required methods ('forward' or 'encode').")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=tuple(errors))
