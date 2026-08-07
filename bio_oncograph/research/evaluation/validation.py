"""Validation utilities for scientific evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str]


class EvaluationValidator:
    """Validator for evaluation metrics and experiments."""

    @staticmethod
    def validate_metrics(metrics: Dict[str, float]) -> ValidationResult:
        """Validate that metrics are well-formed.

        Args:
            metrics: Dictionary of metric values.

        Returns:
            A ValidationResult indicating validity and any errors.
        """
        logger.info("Validating metrics")
        errors = []
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                errors.append(f"Metric {name} has non-numeric value: {value}")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_experiments(runs: List[Any]) -> ValidationResult:
        """Check experiment runs for duplicates and consistency.

        Args:
            runs: List of experiment runs.

        Returns:
            A ValidationResult indicating validity and any errors.
        """
        logger.info("Validating %d experiments", len(runs))
        errors = []
        seen_ids = set()
        for i, run in enumerate(runs):
            run_id = getattr(run, "run_id", f"unknown_{i}")
            if run_id in seen_ids:
                errors.append(f"Duplicate run ID found: {run_id}")
            seen_ids.add(run_id)
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_statistical_input(a: List[float], b: List[float]) -> ValidationResult:
        """Validate input arrays for statistical testing.

        Args:
            a: First array of numerical values.
            b: Second array of numerical values.

        Returns:
            A ValidationResult indicating validity and any errors.
        """
        logger.info("Validating statistical input")
        errors = []
        if len(a) == 0 or len(b) == 0:
            errors.append("One or both input arrays are empty")
        if len(a) != len(b):
            errors.append("Input arrays have mismatched lengths")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
