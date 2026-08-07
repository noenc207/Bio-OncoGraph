"""Federated learning server module."""

import logging
from typing import Tuple

import numpy as np

from bio_oncograph.research.federated.config import FederatedConfig

logger = logging.getLogger(__name__)


class FederatedServer:
    """Server coordinating the federated learning process."""

    def __init__(self, config: FederatedConfig, model_shape: Tuple[int, ...]) -> None:
        """Initializes the federated server.

        Args:
            config: Configuration for the federated learning session.
            model_shape: Shape of the global model parameters.
        """
        self.config = config
        self._model_shape = model_shape
        # Initialize the global model with zeros (or a specific initialization scheme)
        self._global_model = np.zeros(shape=model_shape, dtype=np.float32)
        logger.info("Initialized federated server with model shape %s.", model_shape)

    def get_global_model(self) -> np.ndarray:
        """Retrieves the current global model.

        Returns:
            A numpy array representing the global model parameters.
        """
        return np.copy(self._global_model)

    def update_model(self, aggregated_update: np.ndarray) -> None:
        """Updates the global model with the aggregated update.

        Args:
            aggregated_update: The aggregated update from clients.
        
        Raises:
            ValueError: If the shape of the update does not match the model.
        """
        if aggregated_update.shape != self._model_shape:
            logger.error(
                "Update shape mismatch. Expected %s, got %s.",
                self._model_shape,
                aggregated_update.shape
            )
            raise ValueError(
                f"Update shape {aggregated_update.shape} does not match model "
                f"shape {self._model_shape}."
            )
        
        self._global_model = aggregated_update
        logger.info("Global model updated.")

    def broadcast(self) -> np.ndarray:
        """Broadcasts the global model to the clients.

        Returns:
            The current global model parameters.
        """
        logger.info("Broadcasting global model to clients.")
        return self.get_global_model()
