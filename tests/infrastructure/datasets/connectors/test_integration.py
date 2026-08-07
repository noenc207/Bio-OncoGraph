"""Integration tests for the Dataset Connectors Framework."""

import pytest
import os
import tempfile
from typing import Any, Dict, List

# Simulating imports from the framework
class ConnectorRegistry:
    _connectors = {}
    @classmethod
    def register(cls, name, connector_cls): cls._connectors[name] = connector_cls
    @classmethod
    def get(cls, name): return cls._connectors[name]

class MockPreProcessor:
    def process(self, data: List[Any]) -> List[Any]:
        return [d * 2 for d in data]

class IntegrationConnector:
    def __init__(self, name: str):
        self.name = name
        self.connected = False
        
    def connect(self):
        self.connected = True
        return True
        
    def fetch(self):
        if not self.connected: raise RuntimeError("Not connected")
        return [1, 2, 3, 4, 5]
        
    def get_metadata(self):
        return {"id": self.name, "version": "1.0", "samples": 5}

class FrameworkPipeline:
    """Simulates the full dataset pipeline."""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
    def run_pipeline(self, connector_name: str) -> Dict[str, Any]:
        # 1. Registry
        connector_cls = ConnectorRegistry.get(connector_name)
        connector = connector_cls(connector_name)
        
        # 2. Connect & Fetch
        connector.connect()
        data = connector.fetch()
        metadata = connector.get_metadata()
        
        # 3. Validation (Mocked logic)
        if not data:
            raise ValueError("Empty data fetched")
            
        # 4. Preprocessing
        processor = MockPreProcessor()
        processed_data = processor.process(data)
        
        # 5. Caching
        cache_path = os.path.join(self.cache_dir, f"{connector_name}_v{metadata['version']}.txt")
        with open(cache_path, "w") as f:
            f.write(str(processed_data))
            
        return {
            "metadata": metadata,
            "data": processed_data,
            "cached_at": cache_path
        }

@pytest.fixture(autouse=True)
def setup_registry():
    ConnectorRegistry._connectors.clear()
    ConnectorRegistry.register("integration_test_connector", IntegrationConnector)
    yield
    ConnectorRegistry._connectors.clear()

@pytest.fixture
def temp_cache():
    with tempfile.TemporaryDirectory() as d:
        yield d

def test_full_pipeline_success(temp_cache):
    """Test the complete end-to-end dataset pipeline."""
    pipeline = FrameworkPipeline(temp_cache)
    result = pipeline.run_pipeline("integration_test_connector")
    
    assert result["metadata"]["samples"] == 5
    assert result["data"] == [2, 4, 6, 8, 10]
    assert os.path.exists(result["cached_at"])

def test_pipeline_connector_not_found(temp_cache):
    """Test pipeline failure when connector is not registered."""
    pipeline = FrameworkPipeline(temp_cache)
    with pytest.raises(KeyError):
        pipeline.run_pipeline("missing_connector")

def test_pipeline_data_processing(temp_cache):
    """Test that data is correctly transformed by preprocessor."""
    pipeline = FrameworkPipeline(temp_cache)
    result = pipeline.run_pipeline("integration_test_connector")
    
    # Original data [1,2,3,4,5] -> processor(*2) -> [2,4,6,8,10]
    assert sum(result["data"]) == 30

def test_pipeline_cache_creation(temp_cache):
    """Test that the pipeline correctly caches the results."""
    pipeline = FrameworkPipeline(temp_cache)
    result = pipeline.run_pipeline("integration_test_connector")
    
    with open(result["cached_at"], "r") as f:
        content = f.read()
    assert "[2, 4, 6, 8, 10]" in content

def test_pipeline_metadata_extraction(temp_cache):
    """Test that metadata is properly extracted and returned."""
    pipeline = FrameworkPipeline(temp_cache)
    result = pipeline.run_pipeline("integration_test_connector")
    
    meta = result["metadata"]
    assert meta["id"] == "integration_test_connector"
    assert meta["version"] == "1.0"

def test_pipeline_caching_filename_format(temp_cache):
    """Test the formatting of the cache filename."""
    pipeline = FrameworkPipeline(temp_cache)
    result = pipeline.run_pipeline("integration_test_connector")
    
    assert result["cached_at"].endswith("integration_test_connector_v1.0.txt")

def test_pipeline_multiple_runs(temp_cache):
    """Test running the pipeline multiple times."""
    pipeline = FrameworkPipeline(temp_cache)
    res1 = pipeline.run_pipeline("integration_test_connector")
    res2 = pipeline.run_pipeline("integration_test_connector")
    
    assert res1["data"] == res2["data"]

def test_integration_connector_disconnected_fetch():
    """Test the internal connector logic without the pipeline."""
    conn = IntegrationConnector("test")
    with pytest.raises(RuntimeError, match="Not connected"):
        conn.fetch()

def test_integration_connector_connect_success():
    """Test connector connect logic."""
    conn = IntegrationConnector("test")
    assert conn.connect() is True
    assert conn.connected is True

def test_processor_logic():
    """Test the mock preprocessor directly."""
    p = MockPreProcessor()
    assert p.process([10, 20]) == [20, 40]
