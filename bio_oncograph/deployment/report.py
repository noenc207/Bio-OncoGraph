"""Reporting module for summarizing deployment platform state."""
import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DeploymentReport:
    """Immutable report of the deployment platform's state."""
    deployments_active: int
    models_registered: int
    drift_alerts: int
    health_status: Dict[str, bool]

class ReportGenerator:
    """Generates comprehensive deployment platform reports."""

    def compile(self, active_deployments: int, registered_models: int, active_alerts: int, health_results: Dict[str, bool]) -> DeploymentReport:
        """Compiles the platform state into a report.

        Args:
            active_deployments: Number of currently active deployments.
            registered_models: Total number of registered models.
            active_alerts: Number of current drift alerts.
            health_results: Dictionary mapping component names to health booleans.

        Returns:
            A populated DeploymentReport instance.
        """
        logger.info("Compiling deployment report.")
        return DeploymentReport(
            deployments_active=active_deployments,
            models_registered=registered_models,
            drift_alerts=active_alerts,
            health_status=health_results
        )
