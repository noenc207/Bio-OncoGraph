import logging
from dataclasses import dataclass
from typing import List, Any, Dict

from bio_oncograph.research.federated.report import FederatedReport, ReportGenerator
from bio_oncograph.research.federated.validation import FederatedValidator, ValidationResult

logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    num_rounds: int
    clients_per_round: int
    aggregation_method: str = "fedavg"
    privacy_method: str = "none"

class FederatedLearningEngine:
    """Facade class for the federated learning engine."""
    
    def __init__(self, config: FederatedConfig) -> None:
        """Initializes the FederatedLearningEngine.
        
        Args:
            config: Configuration for the engine.
        """
        self.config = config
        self.clients: List[Any] = []
        self.current_round = 0
        self.metrics: Dict[str, float] = {}
        self.validator = FederatedValidator()
        self.report_generator = ReportGenerator()
        logger.info("Initialized FederatedLearningEngine with config: %s", config)
        
    def register_client(self, client: Any) -> None:
        """Registers a client with the engine.
        
        Args:
            client: The client to register.
        """
        val_result = self.validator.validate_client(client)
        if val_result.is_valid:
            self.clients.append(client)
            logger.info("Registered client. Total clients: %d", len(self.clients))
        else:
            logger.error("Failed to register client: %s", val_result.errors)
            
    def start_round(self) -> int:
        """Starts a new federated learning round.
        
        Returns:
            The current round number.
        """
        self.current_round += 1
        logger.info("Started round %d", self.current_round)
        return self.current_round
        
    def broadcast(self) -> None:
        """Broadcasts the global model to selected clients."""
        logger.info("Broadcasting global model to clients in round %d.", self.current_round)
        
    def aggregate(self) -> None:
        """Aggregates updates from clients."""
        logger.info("Aggregating client updates in round %d.", self.current_round)
        
    def save_checkpoint(self, path: str) -> None:
        """Saves the current engine state to a checkpoint.
        
        Args:
            path: File path to save the checkpoint.
        """
        logger.info("Saving checkpoint to %s", path)
        
    def load_checkpoint(self, path: str) -> None:
        """Loads engine state from a checkpoint.
        
        Args:
            path: File path to load the checkpoint from.
        """
        logger.info("Loading checkpoint from %s", path)
        
    def validate(self) -> ValidationResult:
        """Validates the current state of the engine.
        
        Returns:
            ValidationResult indicating engine validity.
        """
        logger.debug("Validating engine state.")
        return self.validator.validate_round({'round_number': self.current_round})
        
    def report(self) -> FederatedReport:
        """Generates a report of the engine's current state.
        
        Returns:
            A FederatedReport object.
        """
        logger.debug("Generating engine report.")
        return self.report_generator.generate(self)
