"""Explainability outputs module."""

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass(frozen=True)
class ExplainabilityOutput:
    """Output from the explainability engine."""
    scores: np.ndarray
    metadata: Dict[str, Any]
    warnings: List[str]
    version: str


@dataclass(frozen=True)
class UncertaintyOutput:
    """Output from the uncertainty estimation."""
    scores: np.ndarray
    metadata: Dict[str, Any]
    warnings: List[str]
    version: str


@dataclass(frozen=True)
class TrustScoreOutput:
    """Output from the trust score computation."""
    scores: np.ndarray
    aggregated_trust_score: float
    metadata: Dict[str, Any]
    warnings: List[str]
    version: str
