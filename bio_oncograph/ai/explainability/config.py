"""Explainability configuration module."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Configuration for the Explainability Framework.

    Attributes:
        enable_uncertainty: Whether to enable uncertainty estimation.
        enable_explainability: Whether to enable explainability attribution.
        enable_trust_score: Whether to compute a trust score.
        confidence_threshold: Minimum confidence threshold.
        attribution_method: Method for feature attribution.
        calibration_method: Method for confidence calibration.
    """
    enable_uncertainty: bool = True
    enable_explainability: bool = True
    enable_trust_score: bool = True
    confidence_threshold: float = 0.5
    attribution_method: str = 'integrated_gradients_mock'
    calibration_method: str = 'platt'
