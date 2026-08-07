"""Reporting utilities for clinical models."""
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalReport:
    """Frozen dataclass representing a clinical modeling report."""
    endpoint: str
    model_type: str
    num_samples: int
    censoring_rate: float
    risk_groups: int
    calibration_method: str
    c_index: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClinicalReportGenerator:
    """Generator for clinical reports."""

    def generate(self, engine_state: Dict[str, Any]) -> ClinicalReport:
        """Generate a clinical report from the engine state.
        
        Args:
            engine_state: Dictionary containing the internal state of the engine.
            
        Returns:
            A populated ClinicalReport object.
        """
        logger.info("Generating clinical report from engine state.")
        
        config = engine_state.get("config", {})
        metadata = engine_state.get("metadata", {})
        
        report = ClinicalReport(
            endpoint=config.get("endpoint", "Unknown"),
            model_type=config.get("model_type", "Unknown"),
            num_samples=engine_state.get("num_samples", 0),
            censoring_rate=engine_state.get("censoring_rate", 0.0),
            risk_groups=config.get("risk_groups", 0),
            calibration_method=config.get("calibration_method", "None"),
            c_index=engine_state.get("c_index"),
            metadata=metadata,
        )
        
        logger.info("Clinical report generated successfully.")
        return report
