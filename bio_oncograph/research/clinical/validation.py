"""Validation utilities for clinical modeling."""
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)


class ClinicalValidator:
    """Validator for clinical data and predictions."""

    @staticmethod
    def validate_survival_data(times: np.ndarray, events: np.ndarray) -> ValidationResult:
        """Validate survival times and events.
        
        Args:
            times: Array of survival times.
            events: Array of event indicators (binary).
            
        Returns:
            ValidationResult indicating if the data is valid.
        """
        errors = []
        if not isinstance(times, np.ndarray) or not isinstance(events, np.ndarray):
            errors.append("Times and events must be numpy arrays.")
            return ValidationResult(is_valid=False, errors=errors)
            
        if times.shape != events.shape:
            errors.append("Times and events must have the same shape.")
            
        if np.any(np.isnan(times)) or np.any(np.isnan(events)):
            errors.append("Times and events must not contain NaN.")
            
        if np.any(times < 0):
            errors.append("Survival times must be non-negative.")
            
        if not np.array_equal(events, events.astype(bool)):
            errors.append("Events must be binary (0 or 1).")
            
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning("Survival data validation failed: %s", errors)
            
        return ValidationResult(is_valid=is_valid, errors=errors)

    @staticmethod
    def validate_features(features: np.ndarray) -> ValidationResult:
        """Validate clinical features.
        
        Args:
            features: Array of features.
            
        Returns:
            ValidationResult indicating if the features are valid.
        """
        errors = []
        if not isinstance(features, np.ndarray):
            errors.append("Features must be a numpy array.")
            return ValidationResult(is_valid=False, errors=errors)
            
        if len(features.shape) != 2:
            errors.append("Features must be a 2D array.")
            
        if np.any(np.isnan(features)):
            errors.append("Features must not contain NaN.")
            
        if np.any(np.isinf(features)):
            errors.append("Features must not contain Inf.")
            
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning("Features validation failed: %s", errors)
            
        return ValidationResult(is_valid=is_valid, errors=errors)

    @staticmethod
    def validate_risk_scores(scores: np.ndarray) -> ValidationResult:
        """Validate risk scores.
        
        Args:
            scores: Array of risk scores.
            
        Returns:
            ValidationResult indicating if the risk scores are valid.
        """
        errors = []
        if not isinstance(scores, np.ndarray):
            errors.append("Risk scores must be a numpy array.")
            return ValidationResult(is_valid=False, errors=errors)
            
        if np.any(np.isnan(scores)):
            errors.append("Risk scores must not contain NaN.")
            
        if np.any(np.isinf(scores)):
            errors.append("Risk scores must not contain Inf.")
            
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning("Risk scores validation failed: %s", errors)
            
        return ValidationResult(is_valid=is_valid, errors=errors)

    @staticmethod
    def validate_endpoint(endpoint_name: str) -> ValidationResult:
        """Validate clinical endpoint name.
        
        Args:
            endpoint_name: Name of the clinical endpoint.
            
        Returns:
            ValidationResult indicating if the endpoint is supported.
        """
        supported_endpoints = {"OS", "PFS", "DFS", "RFS"}
        errors = []
        
        if endpoint_name not in supported_endpoints:
            errors.append(f"Endpoint '{endpoint_name}' is not supported. Supported endpoints: {supported_endpoints}.")
            
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning("Endpoint validation failed: %s", errors)
            
        return ValidationResult(is_valid=is_valid, errors=errors)
