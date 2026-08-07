import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ClinicalConfig:
    """Configuration for clinical outcome and survival modeling."""
    endpoint: str = 'OS'
    model_type: str = 'cox'
    num_risk_groups: int = 3
    risk_thresholds: Optional[Tuple[float, ...]] = None
    censoring_type: str = 'right'
    calibration_method: str = 'platt'
    time_horizons: Tuple[float, ...] = (12.0, 36.0, 60.0)
    seed: int = 42
    deterministic: bool = True
