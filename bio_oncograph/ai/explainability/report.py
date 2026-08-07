"""Explainability report module."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from bio_oncograph.ai.explainability.outputs import (
    ExplainabilityOutput,
    TrustScoreOutput,
    UncertaintyOutput,
)


@dataclass(frozen=True)
class ExplainabilityReport:
    """Consolidated explainability report."""
    is_valid: bool
    explainability: Optional[ExplainabilityOutput]
    uncertainty: Optional[UncertaintyOutput]
    trust: Optional[TrustScoreOutput]
    metadata: Dict[str, Any]
