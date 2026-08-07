import logging
from dataclasses import dataclass
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FederatedReport:
    """Immutable data class for federated learning reports."""
    total_rounds: int
    active_clients: int
    aggregation_method: str
    privacy_method: str
    total_epsilon: float
    final_metrics: Dict[str, float]

class ReportGenerator:
    """Compiles federated learning engine state into a report."""
    
    def __init__(self) -> None:
        """Initializes the ReportGenerator."""
        logger.info("Initialized ReportGenerator.")
        
    def generate(self, engine: Any) -> FederatedReport:
        """Generates a federated learning report from the engine state.
        
        Args:
            engine: The federated learning engine instance.
            
        Returns:
            A FederatedReport containing the compiled metrics.
        """
        logger.debug("Generating report from engine state.")
        return FederatedReport(
            total_rounds=getattr(engine, 'current_round', 0),
            active_clients=len(getattr(engine, 'clients', [])),
            aggregation_method=getattr(engine, 'aggregation_method', "fedavg"),
            privacy_method=getattr(engine, 'privacy_method', "none"),
            total_epsilon=getattr(engine, 'total_epsilon', 0.0),
            final_metrics=getattr(engine, 'metrics', {})
        )
