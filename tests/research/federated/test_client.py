"""Tests for Federated Clients."""
import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional

class FederatedClient(ABC):
    """Abstract base class for federated clients."""
    @abstractmethod
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Fit the model."""
        pass
    
class HospitalClient(FederatedClient):
    """Hospital client mock."""
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        return parameters, 100, {"type": "hospital"}

class LaboratoryClient(FederatedClient):
    """Laboratory client mock."""
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        return parameters, 50, {"type": "lab"}

class ResearchInstituteClient(FederatedClient):
    """Research institute client mock."""
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        return parameters, 200, {"type": "research"}

class SimulationClient(FederatedClient):
    """Simulation client mock."""
    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        return parameters, 10, {"type": "sim"}

class ClientRegistry:
    """Registry for managing clients."""
    def __init__(self) -> None:
        self.clients: Dict[str, FederatedClient] = {}
        
    def register(self, cid: str, client: FederatedClient) -> None:
        """Register a client."""
        self.clients[cid] = client
        
    def get(self, cid: str) -> Optional[FederatedClient]:
        """Get a client."""
        return self.clients.get(cid)

@pytest.mark.parametrize("client_class, expected_type, expected_num", [
    (HospitalClient, "hospital", 100),
    (LaboratoryClient, "lab", 50),
    (ResearchInstituteClient, "research", 200),
    (SimulationClient, "sim", 10)
])
@pytest.mark.parametrize("shape", [(10,), (5, 5), (100, 10)])
def test_client_fit(client_class: type, expected_type: str, expected_num: int, shape: tuple) -> None:
    """Test client fit."""
    client = client_class()
    params = [np.ones(shape)]
    res_params, num_examples, metrics = client.fit(params)
    assert np.array_equal(res_params[0], params[0])
    assert num_examples == expected_num
    assert metrics["type"] == expected_type

@pytest.mark.parametrize("num_clients", [1, 5, 10, 50])
@pytest.mark.parametrize("client_class", [HospitalClient, LaboratoryClient, ResearchInstituteClient, SimulationClient])
def test_client_registry(num_clients: int, client_class: type) -> None:
    """Test registry."""
    registry = ClientRegistry()
    for i in range(num_clients):
        registry.register(f"client_{i}", client_class())
    
    assert len(registry.clients) == num_clients
    assert isinstance(registry.get("client_0"), client_class)
    assert registry.get("invalid_client") is None
