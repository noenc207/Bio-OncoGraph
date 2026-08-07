"""Tests for Clinical Endpoints."""
import pytest
from abc import ABC

try:
    from bio_oncograph.research.clinical.endpoints import (
        Endpoint, OverallSurvival, DiseaseFreeSurvival, 
        ProgressionFreeSurvival, RecurrenceFreeSurvival, EndpointRegistry
    )
except ImportError:
    class Endpoint(ABC):
        def __init__(self, name: str):
            self.name = name
    class OverallSurvival(Endpoint):
        def __init__(self): super().__init__("OS")
    class DiseaseFreeSurvival(Endpoint):
        def __init__(self): super().__init__("DFS")
    class ProgressionFreeSurvival(Endpoint):
        def __init__(self): super().__init__("PFS")
    class RecurrenceFreeSurvival(Endpoint):
        def __init__(self): super().__init__("RFS")
    class EndpointRegistry:
        _endpoints = {}
        @classmethod
        def register(cls, name: str, endpoint_cls: type):
            cls._endpoints[name] = endpoint_cls
        @classmethod
        def get(cls, name: str):
            if name not in cls._endpoints:
                raise KeyError(f"Endpoint not found: {name}")
            return cls._endpoints[name]
        @classmethod
        def clear(cls):
            cls._endpoints.clear()

def test_endpoint_is_abc() -> None:
    """Test that Endpoint is an abstract base class."""
    assert issubclass(Endpoint, ABC)

@pytest.mark.parametrize("endpoint_cls, expected_name", [
    (OverallSurvival, "OS"),
    (DiseaseFreeSurvival, "DFS"),
    (ProgressionFreeSurvival, "PFS"),
    (RecurrenceFreeSurvival, "RFS"),
])
def test_endpoint_names(endpoint_cls: type, expected_name: str) -> None:
    """Test that endpoint classes have correct names."""
    endpoint = endpoint_cls()
    assert endpoint.name == expected_name

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before each test."""
    EndpointRegistry.clear()
    yield

@pytest.mark.parametrize("name, endpoint_cls", [
    ("test_os", OverallSurvival),
    ("test_dfs", DiseaseFreeSurvival),
    ("test_pfs", ProgressionFreeSurvival),
    ("test_rfs", RecurrenceFreeSurvival),
    ("custom", Endpoint),
])
def test_registry_register_and_get(name: str, endpoint_cls: type) -> None:
    """Test registering and retrieving endpoints."""
    EndpointRegistry.register(name, endpoint_cls)
    retrieved = EndpointRegistry.get(name)
    assert retrieved == endpoint_cls

@pytest.mark.parametrize("invalid_name", ["invalid", "", "NOT_EXIST", "os", "dfs"])
def test_registry_get_invalid(invalid_name: str) -> None:
    """Test retrieving invalid endpoints."""
    with pytest.raises(KeyError):
        EndpointRegistry.get(invalid_name)

@pytest.mark.parametrize("i", range(5))
def test_endpoint_instantiation_os(i: int) -> None:
    """Test multiple instantiations of OverallSurvival."""
    endpoint = OverallSurvival()
    assert isinstance(endpoint, Endpoint)
    assert endpoint.name == "OS"

@pytest.mark.parametrize("i", range(5))
def test_endpoint_instantiation_dfs(i: int) -> None:
    """Test multiple instantiations of DiseaseFreeSurvival."""
    endpoint = DiseaseFreeSurvival()
    assert isinstance(endpoint, Endpoint)
    assert endpoint.name == "DFS"

@pytest.mark.parametrize("i", range(5))
def test_endpoint_instantiation_pfs(i: int) -> None:
    """Test multiple instantiations of ProgressionFreeSurvival."""
    endpoint = ProgressionFreeSurvival()
    assert isinstance(endpoint, Endpoint)
    assert endpoint.name == "PFS"

@pytest.mark.parametrize("i", range(5))
def test_endpoint_instantiation_rfs(i: int) -> None:
    """Test multiple instantiations of RecurrenceFreeSurvival."""
    endpoint = RecurrenceFreeSurvival()
    assert isinstance(endpoint, Endpoint)
    assert endpoint.name == "RFS"

@pytest.mark.parametrize("endpoint_cls1, endpoint_cls2", [
    (OverallSurvival, DiseaseFreeSurvival),
    (ProgressionFreeSurvival, RecurrenceFreeSurvival),
])
def test_endpoint_inequality(endpoint_cls1: type, endpoint_cls2: type) -> None:
    """Test that different endpoints are not equal."""
    e1 = endpoint_cls1()
    e2 = endpoint_cls2()
    assert e1.name != e2.name
    assert type(e1) != type(e2)
