"""Tests for the ConnectorRegistry."""

import pytest
from typing import Dict, Type, Any

# Mock classes for testing
class BaseConnector:
    def __init__(self, name: str):
        self.name = name

class MockConnectorA(BaseConnector): pass
class MockConnectorB(BaseConnector): pass

class ConnectorRegistry:
    """Registry for managing dataset connectors."""
    
    _connectors: Dict[str, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, name: str, connector_cls: Type[BaseConnector]) -> None:
        if name in cls._connectors:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._connectors[name] = connector_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseConnector]:
        if name not in cls._connectors:
            raise KeyError(f"Connector '{name}' not found in registry.")
        return cls._connectors[name]

    @classmethod
    def list_connectors(cls) -> list[str]:
        return list(cls._connectors.keys())

    @classmethod
    def clear(cls) -> None:
        cls._connectors.clear()

@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure registry is empty before and after each test."""
    ConnectorRegistry.clear()
    yield
    ConnectorRegistry.clear()

def test_registry_initial_state():
    """Test that registry starts empty."""
    assert ConnectorRegistry.list_connectors() == []

def test_register_connector():
    """Test registering a valid connector."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    assert "mock_a" in ConnectorRegistry.list_connectors()

def test_register_duplicate_connector():
    """Test that registering a duplicate connector raises ValueError."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    with pytest.raises(ValueError, match="already registered"):
        ConnectorRegistry.register("mock_a", MockConnectorB)

def test_get_connector_success():
    """Test retrieving a registered connector."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    cls = ConnectorRegistry.get("mock_a")
    assert cls is MockConnectorA

def test_get_missing_connector():
    """Test retrieving an unregistered connector raises KeyError."""
    with pytest.raises(KeyError, match="not found in registry"):
        ConnectorRegistry.get("missing")

def test_list_multiple_connectors():
    """Test listing multiple registered connectors."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    ConnectorRegistry.register("mock_b", MockConnectorB)
    assert set(ConnectorRegistry.list_connectors()) == {"mock_a", "mock_b"}

def test_clear_registry():
    """Test clearing the registry."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    ConnectorRegistry.clear()
    assert ConnectorRegistry.list_connectors() == []

def test_register_after_clear():
    """Test registering after clearing works."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    ConnectorRegistry.clear()
    ConnectorRegistry.register("mock_a", MockConnectorB)
    assert ConnectorRegistry.get("mock_a") is MockConnectorB

def test_register_different_types():
    """Test registering different types of connectors."""
    ConnectorRegistry.register("type1", MockConnectorA)
    ConnectorRegistry.register("type2", MockConnectorA)
    assert len(ConnectorRegistry.list_connectors()) == 2

def test_registry_is_singleton_like():
    """Test that state is shared across classes if instantiated (though it shouldn't be)."""
    assert len(ConnectorRegistry._connectors) == 0

def test_registry_get_returns_class_not_instance():
    """Test that get returns the class, not an instance."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    cls = ConnectorRegistry.get("mock_a")
    assert isinstance(cls, type)
    assert issubclass(cls, BaseConnector)

def test_instantiate_from_registry():
    """Test that retrieved classes can be instantiated."""
    ConnectorRegistry.register("mock_a", MockConnectorA)
    cls = ConnectorRegistry.get("mock_a")
    instance = cls(name="test")
    assert instance.name == "test"
    assert isinstance(instance, MockConnectorA)
