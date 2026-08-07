import logging
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationResult:
    """Result of a censoring validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class CensoringValidator:
    """Validates and processes censored survival data."""

    def validate_right_censoring(self, times: np.ndarray, events: np.ndarray) -> ValidationResult:
        """Validates right-censored data."""
        errors = []
        if np.any(times < 0):
            errors.append("Negative survival times found.")
        if not set(np.unique(events)).issubset({0, 1}):
            errors.append("Events must be binary (0 or 1).")
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors)

    def validate_left_censoring(self, times: np.ndarray, events: np.ndarray) -> ValidationResult:
        """Validates left-censored data."""
        errors = []
        if np.any(times < 0):
            errors.append("Negative survival times found.")
        if not set(np.unique(events)).issubset({0, 1}):
            errors.append("Events must be binary (0 or 1).")
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors)

    def validate_interval_censoring(self, times_lower: np.ndarray, times_upper: np.ndarray, events: np.ndarray) -> ValidationResult:
        """Validates interval-censored data."""
        errors = []
        if np.any(times_lower > times_upper):
            errors.append("Lower time bounds must be <= upper time bounds.")
        if np.any(times_lower < 0):
            errors.append("Negative survival times found in lower bounds.")
        if not set(np.unique(events)).issubset({0, 1, 2, 3}):
            errors.append("Events must be valid interval censoring codes.")
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors)

    def compute_censoring_rate(self, events: np.ndarray) -> float:
        """Computes the proportion of censored individuals."""
        if len(events) == 0:
            return 0.0
        return float(np.mean(events == 0))

    def kaplan_meier_estimate(self, times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the Kaplan-Meier estimate of the survival curve."""
        if len(times) == 0:
            return np.array([]), np.array([])
        sort_idx = np.argsort(times)
        sorted_times = times[sort_idx]
        sorted_events = events[sort_idx]

        unique_times, time_counts = np.unique(sorted_times, return_counts=True)
        n_at_risk = np.zeros_like(unique_times, dtype=int)
        n_events = np.zeros_like(unique_times, dtype=int)

        current_at_risk = len(times)
        for i, t in enumerate(unique_times):
            n_at_risk[i] = current_at_risk
            mask = (sorted_times == t)
            n_events[i] = np.sum(sorted_events[mask])
            current_at_risk -= time_counts[i]

        survival_probs = np.cumprod(1.0 - n_events / n_at_risk)
        return unique_times, survival_probs
