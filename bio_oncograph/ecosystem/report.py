import logging
import time
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EcosystemReport:
    """Summarizes the state and outputs of the scientific ecosystem.
    
    Attributes:
        experiments_count: Number of experiments run.
        leaderboard_entries: Number of entries on the leaderboard.
        model_cards_generated: Number of generated model cards.
        dataset_cards_generated: Number of generated dataset cards.
        publications_exported: Number of publications exported.
        artifacts_packaged: Number of packaged artifacts.
        reproducibility_verified: Boolean indicating if all are verified.
        timestamp: Time the report was generated.
    """
    experiments_count: int
    leaderboard_entries: int
    model_cards_generated: int
    dataset_cards_generated: int
    publications_exported: int
    artifacts_packaged: int
    reproducibility_verified: bool
    timestamp: float

class EcosystemReportGenerator:
    """Generates ecosystem reports."""
    
    def compile_report(self, platform_state: Dict[str, Any]) -> EcosystemReport:
        """Compiles a report from the platform state.
        
        Args:
            platform_state: Dictionary containing current counts and states.
            
        Returns:
            An EcosystemReport instance.
        """
        logger.info("Compiling ecosystem report")
        return EcosystemReport(
            experiments_count=platform_state.get("experiments_count", 0),
            leaderboard_entries=platform_state.get("leaderboard_entries", 0),
            model_cards_generated=platform_state.get("model_cards_generated", 0),
            dataset_cards_generated=platform_state.get("dataset_cards_generated", 0),
            publications_exported=platform_state.get("publications_exported", 0),
            artifacts_packaged=platform_state.get("artifacts_packaged", 0),
            reproducibility_verified=platform_state.get("reproducibility_verified", True),
            timestamp=time.time()
        )
        
    def to_markdown(self, report: EcosystemReport) -> str:
        """Converts an EcosystemReport to a Markdown string.
        
        Args:
            report: The EcosystemReport to format.
            
        Returns:
            A Markdown formatted string.
        """
        logger.info("Converting report to Markdown")
        lines = [
            "# Scientific Ecosystem Report",
            f"**Generated on:** {report.timestamp}",
            "",
            "## Summary Statistics",
            f"- **Experiments Run:** {report.experiments_count}",
            f"- **Leaderboard Entries:** {report.leaderboard_entries}",
            f"- **Model Cards Generated:** {report.model_cards_generated}",
            f"- **Dataset Cards Generated:** {report.dataset_cards_generated}",
            f"- **Publications Exported:** {report.publications_exported}",
            f"- **Artifacts Packaged:** {report.artifacts_packaged}",
            f"- **Reproducibility Verified:** {'Yes' if report.reproducibility_verified else 'No'}"
        ]
        return "\n".join(lines)
