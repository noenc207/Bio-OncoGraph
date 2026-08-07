"""Reporting structures for multi-modal learning engine."""
from dataclasses import dataclass
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EngineReport:
    """Report detailing the configuration and state of the multi-modal engine.

    Attributes:
        active_modalities: List of active modality names.
        fusion_strategy: String identifier for the fusion strategy in use.
        alignment_strategy: String identifier for the alignment strategy in use.
        missing_strategy: String identifier for how missing modalities are handled.
        total_processed: Total number of records processed by the engine.
        metadata: Additional metadata about the engine.
    """
    active_modalities: List[str]
    fusion_strategy: str
    alignment_strategy: str
    missing_strategy: str
    total_processed: int
    metadata: Dict[str, Any]

class ReportGenerator:
    """Generates engine reports based on current engine state."""

    def __init__(self) -> None:
        """Initialize the report generator."""
        self.logger = logging.getLogger(__name__ + ".ReportGenerator")

    def generate(self, 
                 active_modalities: List[str],
                 fusion_strategy: str,
                 alignment_strategy: str,
                 missing_strategy: str,
                 total_processed: int,
                 metadata: Dict[str, Any]) -> EngineReport:
        """Generate an EngineReport from the provided state parameters.

        Args:
            active_modalities: List of active modality names.
            fusion_strategy: String identifier for the fusion strategy in use.
            alignment_strategy: String identifier for the alignment strategy in use.
            missing_strategy: String identifier for how missing modalities are handled.
            total_processed: Total number of records processed by the engine.
            metadata: Additional metadata about the engine.

        Returns:
            A populated EngineReport object.
        """
        report = EngineReport(
            active_modalities=list(active_modalities),
            fusion_strategy=fusion_strategy,
            alignment_strategy=alignment_strategy,
            missing_strategy=missing_strategy,
            total_processed=total_processed,
            metadata=dict(metadata)
        )
        self.logger.info("Generated new engine report.")
        return report
