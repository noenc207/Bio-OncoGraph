"""Tests for the base connector interface."""

import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseConnector(ABC):
    """Abstract base class for dataset connectors."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the data source."""
        pass

    @abstractmethod
    def fetch(self) -> List[Any]:
        """Fetch data from the source."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Retrieve metadata for the dataset."""
        pass
        
    def disconnect(self) -> bool:
        """Disconnect from the data source."""
        self.is_connected = False
        return True

class IncompleteConnector(BaseConnector):
    """Connector missing implementation of abstract methods."""
    pass

class DummyConnector(BaseConnector):
    """Concrete implementation for testing."""
    def connect(self) -> bool:
        self.is_connected = True
        return True

    def fetch(self) -> List[Any]:
        if not self.is_connected:
            raise RuntimeError("Not connected")
        return [1, 2, 3]

    def get_metadata(self) -> Dict[str, Any]:
        return {"size": 3, "type": "dummy"}

def test_base_connector_instantiation_fails():
    """Test that BaseConnector cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseConnector"):
        BaseConnector(name="base")

def test_incomplete_connector_instantiation_fails():
    """Test that incomplete connectors cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteConnector"):
        IncompleteConnector(name="incomplete")

def test_dummy_connector_initialization():
    """Test successful initialization of a concrete connector."""
    connector = DummyConnector(name="dummy")
    assert connector.name == "dummy"
    assert connector.config == {}
    assert connector.is_connected is False

def test_dummy_connector_with_config():
    """Test initialization with configuration."""
    connector = DummyConnector(name="dummy", config={"api_key": "123"})
    assert connector.config["api_key"] == "123"

def test_dummy_connector_connect():
    """Test connect method implementation."""
    connector = DummyConnector(name="dummy")
    assert connector.connect() is True
    assert connector.is_connected is True

def test_dummy_connector_fetch_without_connect():
    """Test fetch before connecting raises error."""
    connector = DummyConnector(name="dummy")
    with pytest.raises(RuntimeError, match="Not connected"):
        connector.fetch()

def test_dummy_connector_fetch_success():
    """Test successful data fetching."""
    connector = DummyConnector(name="dummy")
    connector.connect()
    assert connector.fetch() == [1, 2, 3]

def test_dummy_connector_get_metadata():
    """Test metadata retrieval."""
    connector = DummyConnector(name="dummy")
    assert connector.get_metadata() == {"size": 3, "type": "dummy"}

def test_dummy_connector_disconnect():
    """Test disconnection logic."""
    connector = DummyConnector(name="dummy")
    connector.connect()
    assert connector.disconnect() is True
    assert connector.is_connected is False

def test_dummy_connector_reconnect():
    """Test reconnecting after disconnect."""
    connector = DummyConnector(name="dummy")
    connector.connect()
    connector.disconnect()
    assert connector.connect() is True
    assert connector.is_connected is True

def test_dummy_connector_state_isolation():
    """Test that multiple instances maintain isolated state."""
    c1 = DummyConnector(name="c1")
    c2 = DummyConnector(name="c2")
    c1.connect()
    assert c1.is_connected is True
    assert c2.is_connected is False

def test_dummy_connector_name_immutability():
    """Test name attribution."""
    connector = DummyConnector(name="test")
    assert getattr(connector, "name") == "test"
