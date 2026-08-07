import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

from bio_oncograph.research.clinical.risk import RiskOutput

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SurvivalOutput:
    """Output containing survival probability estimates."""
    survival_probabilities: np.ndarray
    time_horizons: Tuple[float, ...]
    metadata: Dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"


@dataclass(frozen=True)
class ClinicalPredictionOutput:
    """Aggregated clinical prediction output."""
    endpoint: str
    risk_output: RiskOutput
    survival_output: SurvivalOutput
    calibration_applied: bool
    metadata: Dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"
