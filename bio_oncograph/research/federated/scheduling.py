import logging
import random
from typing import List, Any, Dict

logger = logging.getLogger(__name__)

class RoundScheduler:
    """Scheduler for managing federated learning rounds and clients."""
    
    def __init__(self) -> None:
        """Initializes the RoundScheduler."""
        logger.info("Initialized RoundScheduler.")
        
    def sample_clients(self, available_clients: List[Any], num_clients: int, seed: int = None) -> List[Any]:
        """Samples a subset of clients uniformly at random.
        
        Args:
            available_clients: List of all available clients.
            num_clients: Number of clients to sample.
            seed: Random seed for reproducible sampling.
            
        Returns:
            List of sampled clients.
            
        Raises:
            ValueError: If num_clients is greater than available_clients.
        """
        if num_clients > len(available_clients):
            raise ValueError("Requested more clients than available.")
            
        if seed is not None:
            random.seed(seed)
            
        return random.sample(available_clients, num_clients)
        
    def handle_timeout(self, client_tasks: Dict[str, float], timeout: float) -> List[str]:
        """Identifies tasks that completed within the timeout.
        
        Args:
            client_tasks: Dictionary mapping client IDs to completion times.
            timeout: Maximum allowed time in seconds.
            
        Returns:
            List of client IDs whose tasks completed within the timeout.
        """
        completed_clients = [
            client_id for client_id, completion_time in client_tasks.items()
            if completion_time <= timeout
        ]
        
        stragglers = len(client_tasks) - len(completed_clients)
        if stragglers > 0:
            logger.warning("Dropped %d straggler tasks due to timeout.", stragglers)
            
        return completed_clients
