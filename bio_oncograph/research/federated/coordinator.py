import logging

from bio_oncograph.research.federated.engine import FederatedLearningEngine

logger = logging.getLogger(__name__)

class FederationCoordinator:
    """Orchestrates the federated learning loop."""
    
    def __init__(self) -> None:
        """Initializes the FederationCoordinator."""
        logger.info("Initialized FederationCoordinator.")
        
    def run_federation(self, engine: FederatedLearningEngine) -> None:
        """Runs the entire federated learning loop.
        
        Args:
            engine: The FederatedLearningEngine instance to orchestrate.
        """
        logger.info("Starting federation run for %d rounds.", engine.config.num_rounds)
        
        for round_num in range(1, engine.config.num_rounds + 1):
            engine.start_round()
            
            # Step 1: Broadcast
            logger.debug("Round %d: Broadcasting model.", round_num)
            engine.broadcast()
            
            # Step 2: Train (mocked via client local processing, skipped here)
            logger.debug("Round %d: Waiting for client training.", round_num)
            
            # Step 3: Collect (mocked)
            logger.debug("Round %d: Collecting updates (handling stragglers/privacy).", round_num)
            
            # Step 4: Aggregate
            logger.debug("Round %d: Aggregating updates.", round_num)
            engine.aggregate()
            
            # Validate round
            val_result = engine.validate()
            if not val_result.is_valid:
                logger.error("Round %d validation failed: %s", round_num, val_result.errors)
                break
                
            logger.info("Completed round %d successfully.", round_num)
            
        final_report = engine.report()
        logger.info("Federation run finished. Final report: %s", final_report)
