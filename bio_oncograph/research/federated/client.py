"""Federated learning client representations."""

import abc
import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class FederatedClient(abc.ABC):
    """Abstract base class for a federated learning client."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Retrieves the client identifier.

        Returns:
            The client identifier as a string.
        """
        pass

    @abc.abstractmethod
    def train(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Trains the model locally on the client data.

        Args:
            global_model: The global model parameters from the server.

        Returns:
            A dictionary containing the local update and metadata (e.g., sample count).
        """
        pass

    @abc.abstractmethod
    def evaluate(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Evaluates the global model locally on the client data.

        Args:
            global_model: The global model parameters from the server.

        Returns:
            A dictionary containing evaluation metrics.
        """
        pass


class HospitalClient(FederatedClient):
    """Mock client representing a hospital."""

    def __init__(self, client_id: str, sample_count: int = 100) -> None:
        """Initializes the HospitalClient.

        Args:
            client_id: Unique identifier for the client.
            sample_count: Number of simulated local samples.
        """
        self._client_id = client_id
        self._sample_count = sample_count

    def get_id(self) -> str:
        """Retrieves the client identifier."""
        return self._client_id

    def train(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local training."""
        logger.info("Hospital client %s training.", self._client_id)
        local_update = global_model + np.random.normal(scale=0.01, size=global_model.shape)
        return {"update": local_update, "sample_count": self._sample_count}

    def evaluate(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local evaluation."""
        logger.info("Hospital client %s evaluating.", self._client_id)
        return {"loss": float(np.random.uniform(0.1, 1.0))}


class LaboratoryClient(FederatedClient):
    """Mock client representing a laboratory."""

    def __init__(self, client_id: str, sample_count: int = 50) -> None:
        """Initializes the LaboratoryClient.

        Args:
            client_id: Unique identifier for the client.
            sample_count: Number of simulated local samples.
        """
        self._client_id = client_id
        self._sample_count = sample_count

    def get_id(self) -> str:
        """Retrieves the client identifier."""
        return self._client_id

    def train(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local training."""
        logger.info("Laboratory client %s training.", self._client_id)
        local_update = global_model + np.random.normal(scale=0.02, size=global_model.shape)
        return {"update": local_update, "sample_count": self._sample_count}

    def evaluate(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local evaluation."""
        logger.info("Laboratory client %s evaluating.", self._client_id)
        return {"loss": float(np.random.uniform(0.2, 1.2))}


class ResearchInstituteClient(FederatedClient):
    """Mock client representing a research institute."""

    def __init__(self, client_id: str, sample_count: int = 200) -> None:
        """Initializes the ResearchInstituteClient.

        Args:
            client_id: Unique identifier for the client.
            sample_count: Number of simulated local samples.
        """
        self._client_id = client_id
        self._sample_count = sample_count

    def get_id(self) -> str:
        """Retrieves the client identifier."""
        return self._client_id

    def train(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local training."""
        logger.info("Research institute client %s training.", self._client_id)
        local_update = global_model + np.random.normal(scale=0.005, size=global_model.shape)
        return {"update": local_update, "sample_count": self._sample_count}

    def evaluate(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local evaluation."""
        logger.info("Research institute client %s evaluating.", self._client_id)
        return {"loss": float(np.random.uniform(0.05, 0.8))}


class SimulationClient(FederatedClient):
    """Mock client representing a pure simulation."""

    def __init__(self, client_id: str, sample_count: int = 1000) -> None:
        """Initializes the SimulationClient.

        Args:
            client_id: Unique identifier for the client.
            sample_count: Number of simulated local samples.
        """
        self._client_id = client_id
        self._sample_count = sample_count

    def get_id(self) -> str:
        """Retrieves the client identifier."""
        return self._client_id

    def train(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local training."""
        logger.info("Simulation client %s training.", self._client_id)
        local_update = global_model + np.random.normal(scale=0.001, size=global_model.shape)
        return {"update": local_update, "sample_count": self._sample_count}

    def evaluate(self, global_model: np.ndarray) -> Dict[str, Any]:
        """Performs simulated local evaluation."""
        logger.info("Simulation client %s evaluating.", self._client_id)
        return {"loss": float(np.random.uniform(0.01, 0.5))}


class ClientRegistry:
    """Manages the registration and discovery of federated clients."""

    def __init__(self) -> None:
        """Initializes an empty client registry."""
        self._clients: Dict[str, FederatedClient] = {}

    def register(self, client: FederatedClient) -> None:
        """Registers a client.

        Args:
            client: The client instance to register.

        Raises:
            ValueError: If a client with the same ID is already registered.
        """
        client_id = client.get_id()
        if client_id in self._clients:
            logger.error("Client ID %s already registered.", client_id)
            raise ValueError(f"Client {client_id} already registered.")
        self._clients[client_id] = client
        logger.info("Registered client %s.", client_id)

    def unregister(self, client_id: str) -> None:
        """Unregisters a client.

        Args:
            client_id: The ID of the client to unregister.

        Raises:
            KeyError: If the client is not found.
        """
        if client_id not in self._clients:
            logger.error("Client ID %s not found for unregistration.", client_id)
            raise KeyError(f"Client {client_id} not found.")
        del self._clients[client_id]
        logger.info("Unregistered client %s.", client_id)

    def get_client(self, client_id: str) -> FederatedClient:
        """Retrieves a client by its ID.

        Args:
            client_id: The ID of the client.

        Returns:
            The client instance.

        Raises:
            KeyError: If the client is not found.
        """
        if client_id not in self._clients:
            logger.error("Client ID %s not found.", client_id)
            raise KeyError(f"Client {client_id} not found.")
        return self._clients[client_id]

    def get_all_clients(self) -> List[FederatedClient]:
        """Retrieves all registered clients.

        Returns:
            A list of all client instances.
        """
        return list(self._clients.values())
