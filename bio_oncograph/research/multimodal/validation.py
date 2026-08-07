"""Validation components for multi-modal learning engine."""
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationReport:
    """Report generated after validating multi-modal data and states.

    Attributes:
        is_valid: Boolean indicating whether validation passed.
        errors: Tuple of error messages if validation failed.
        warnings: Tuple of warning messages.
    """
    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]

class MultimodalValidator:
    """Validates multi-modal data, embeddings, and states."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__ + ".MultimodalValidator")

    def validate_modality(self, name: str, embedding: np.ndarray) -> List[str]:
        """Validate a single modality embedding.

        Args:
            name: The name of the modality.
            embedding: The embedding array to validate.

        Returns:
            A list of error strings, empty if validation passes.
        """
        errors = []
        if not isinstance(embedding, np.ndarray):
            errors.append(f"Modality '{name}' embedding is not a numpy array.")
            return errors
        if np.isnan(embedding).any():
            errors.append(f"Modality '{name}' embedding contains NaN values.")
        if np.isinf(embedding).any():
            errors.append(f"Modality '{name}' embedding contains Inf values.")
        if embedding.ndim == 0:
            errors.append(f"Modality '{name}' embedding has 0 dimensions.")
        return errors

    def validate_fusion(self, fused_emb: Any) -> List[str]:
        """Validate the fused embedding.

        Args:
            fused_emb: The fused embedding to validate.

        Returns:
            A list of error strings, empty if validation passes.
        """
        errors = []
        if not hasattr(fused_emb, "embedding") or not isinstance(fused_emb.embedding, np.ndarray):
            errors.append("Fused embedding does not have a valid numpy array 'embedding' attribute.")
            return errors
        if np.isnan(fused_emb.embedding).any():
            errors.append("Fused embedding contains NaN values.")
        if np.isinf(fused_emb.embedding).any():
            errors.append("Fused embedding contains Inf values.")
        return errors

    def validate_metadata(self, meta: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate metadata against required keys.

        Args:
            meta: The metadata dictionary.
            required_keys: List of keys that must be present in the metadata.

        Returns:
            A list of error strings, empty if validation passes.
        """
        errors = []
        if not isinstance(meta, dict):
            errors.append("Metadata must be a dictionary.")
            return errors
        for key in required_keys:
            if key not in meta:
                errors.append(f"Missing required key '{key}' in metadata.")
        return errors

    def validate_alignment(self, aligned_embs: Dict[str, np.ndarray]) -> List[str]:
        """Validate that aligned embeddings are compatible.

        Args:
            aligned_embs: Dictionary mapping modality names to aligned embeddings.

        Returns:
            A list of error strings, empty if validation passes.
        """
        errors = []
        if not aligned_embs:
            errors.append("Aligned embeddings dictionary is empty.")
            return errors
        
        shapes = [emb.shape for emb in aligned_embs.values() if isinstance(emb, np.ndarray)]
        if not shapes:
            errors.append("No valid numpy arrays found in aligned embeddings.")
            return errors
        
        first_shape = shapes[0]
        # Check if the first dimension (batch size) is consistent.
        for name, emb in aligned_embs.items():
            if not isinstance(emb, np.ndarray):
                errors.append(f"Aligned embedding for '{name}' is not a numpy array.")
            elif emb.shape[0] != first_shape[0]:
                errors.append(f"Inconsistent batch size for modality '{name}': {emb.shape[0]} != {first_shape[0]}")
        return errors
