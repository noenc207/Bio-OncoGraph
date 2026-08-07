"""Tests for foundation model adapters."""
import pytest
from typing import Type
import numpy as np

class FoundationAdapter:
    """Abstract base class for foundation adapters."""
    def adapt(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class PathologyAdapter(FoundationAdapter):
    """Adapter for pathology images."""
    def adapt(self, x: np.ndarray) -> np.ndarray:
        return np.array(x) * 2

class VisionTransformerAdapter(FoundationAdapter):
    """Adapter for vision transformers."""
    def adapt(self, x: np.ndarray) -> np.ndarray:
        return np.array(x) + 1

class GraphFoundationAdapter(FoundationAdapter):
    """Adapter for graph structures."""
    def adapt(self, x: np.ndarray) -> np.ndarray:
        return np.array(x) - 1

class AdapterRegistry:
    """Registry for foundation adapters."""
    _registry = {}
    
    @classmethod
    def register(cls, name: str, adapter: Type[FoundationAdapter]) -> None:
        """Register an adapter class."""
        cls._registry[name] = adapter
        
    @classmethod
    def get(cls, name: str) -> FoundationAdapter:
        """Get an adapter instance by name."""
        return cls._registry[name]()


@pytest.mark.parametrize("adapter_cls", [PathologyAdapter, VisionTransformerAdapter, GraphFoundationAdapter])
@pytest.mark.parametrize("input_shape", [(3, 224, 224), (1, 256, 256), (3, 512, 512), (3, 1024, 1024)])
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_adapter_processing(adapter_cls: Type[FoundationAdapter], input_shape: tuple, batch_size: int) -> None:
    """Test processing of inputs by different adapters."""
    adapter = adapter_cls()
    dummy_input = np.random.rand(batch_size, *input_shape)
    output = adapter.adapt(dummy_input)
    assert output.shape == dummy_input.shape
    assert isinstance(output, np.ndarray)

@pytest.mark.parametrize("name", ["path", "vit", "graph", "custom_1", "custom_2"])
@pytest.mark.parametrize("adapter_cls", [PathologyAdapter, VisionTransformerAdapter, GraphFoundationAdapter])
def test_adapter_registry(name: str, adapter_cls: Type[FoundationAdapter]) -> None:
    """Test registration and retrieval from AdapterRegistry."""
    AdapterRegistry.register(name, adapter_cls)
    adapter = AdapterRegistry.get(name)
    assert isinstance(adapter, adapter_cls)
