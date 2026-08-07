"""Aggregation strategies for federated learning."""

import abc
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class AggregationStrategy(abc.ABC):
    """Abstract base class for model aggregation strategies."""

    @abc.abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregates client model updates into a single global update.

        Args:
            client_updates: A list of dictionaries, where each dict contains an
                'update' (numpy array) and potentially other metadata like
                'sample_count'.

        Returns:
            The aggregated global update as a numpy array.
        """
        pass


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy.
    
    Computes a weighted average of client updates based on their sample counts.
    """

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregates client updates using weighted average.
        
        Args:
            client_updates: List of dictionaries containing 'update' and 'sample_count'.
            
        Returns:
            The weighted average of the updates.
            
        Raises:
            ValueError: If the list of client updates is empty or malformed.
        """
        if not client_updates:
            logger.error("Cannot aggregate empty client updates list.")
            raise ValueError("No client updates provided.")

        total_samples = sum(update.get("sample_count", 1) for update in client_updates)
        if total_samples <= 0:
            logger.error("Total samples must be positive, got %d.", total_samples)
            raise ValueError("Invalid total sample count.")

        first_update = client_updates[0]["update"]
        aggregated_model = np.zeros_like(first_update, dtype=float)

        for update_dict in client_updates:
            update = update_dict["update"]
            weight = update_dict.get("sample_count", 1) / total_samples
            aggregated_model += weight * update

        logger.info("Successfully completed FedAvg aggregation.")
        return aggregated_model.astype(first_update.dtype)


class FedProx(AggregationStrategy):
    """Federated Proximal (FedProx) aggregation strategy.
    
    A mock implementation that performs a simple average. In a full implementation,
    the proximal term is added during client local training, and the server
    performs a standard or weighted average.
    """

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregates client updates using simple average.
        
        Args:
            client_updates: List of dictionaries containing 'update'.
            
        Returns:
            The simple average of the updates.
            
        Raises:
            ValueError: If the list of client updates is empty.
        """
        if not client_updates:
            logger.error("Cannot aggregate empty client updates list.")
            raise ValueError("No client updates provided.")

        num_clients = len(client_updates)
        first_update = client_updates[0]["update"]
        aggregated_model = np.zeros_like(first_update, dtype=float)

        for update_dict in client_updates:
            aggregated_model += update_dict["update"]

        aggregated_model /= num_clients
        logger.info("Successfully completed FedProx-style simple average aggregation.")
        return aggregated_model.astype(first_update.dtype)


class AdaptiveAggregation(AggregationStrategy):
    """Adaptive aggregation strategy (mock FedOpt).
    
    Simulates an adaptive learning rate at the server level.
    """

    def __init__(self, server_learning_rate: float = 1.0) -> None:
        """Initializes the adaptive aggregation.
        
        Args:
            server_learning_rate: The server-level learning rate.
        """
        self._lr = server_learning_rate

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregates client updates with a simulated server learning rate.
        
        Args:
            client_updates: List of dictionaries containing 'update'.
            
        Returns:
            The adaptively aggregated update.
            
        Raises:
            ValueError: If the list of client updates is empty.
        """
        if not client_updates:
            logger.error("Cannot aggregate empty client updates list.")
            raise ValueError("No client updates provided.")

        # Simplified logic: average them and scale by learning rate
        num_clients = len(client_updates)
        first_update = client_updates[0]["update"]
        aggregated_model = np.zeros_like(first_update, dtype=float)

        for update_dict in client_updates:
            aggregated_model += update_dict["update"]

        aggregated_model = (aggregated_model / num_clients) * self._lr
        logger.info(
            "Successfully completed Adaptive aggregation with learning rate %f.",
            self._lr
        )
        return aggregated_model.astype(first_update.dtype)


class RobustAggregation(AggregationStrategy):
    """Robust aggregation strategy using median.
    
    Resilient to Byzantine faults or severe outliers.
    """

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregates client updates using a coordinate-wise median.
        
        Args:
            client_updates: List of dictionaries containing 'update'.
            
        Returns:
            The median of the updates.
            
        Raises:
            ValueError: If the list of client updates is empty.
        """
        if not client_updates:
            logger.error("Cannot aggregate empty client updates list.")
            raise ValueError("No client updates provided.")

        updates = np.array([update_dict["update"] for update_dict in client_updates])
        aggregated_model = np.median(updates, axis=0)
        
        logger.info("Successfully completed Robust median-based aggregation.")
        return aggregated_model
