import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional

# ==============================================================================
# MOCKS
# ==============================================================================
class ModalityEncoder(ABC):
    """Abstract base class for modality encoders."""
    @abstractmethod
    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode the input modality data."""
        pass

class HistopathologyEncoder(ModalityEncoder):
    def encode(self, data: np.ndarray) -> np.ndarray:
        return data.mean(axis=(1, 2)) if data.ndim == 4 else data

class ClinicalEncoder(ModalityEncoder):
    def encode(self, data: np.ndarray) -> np.ndarray:
        return np.log1p(np.abs(data))

class GenomicsEncoder(ModalityEncoder):
    def encode(self, data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, 1.0, 0.0)

class ModalityRegistry:
    """Registry to manage encoders."""
    def __init__(self) -> None:
        self._encoders: Dict[str, Type[ModalityEncoder]] = {}

    def register(self, name: str, encoder_cls: Type[ModalityEncoder]) -> None:
        if name in self._encoders:
            raise ValueError(f"Encoder for {name} is already registered.")
        self._encoders[name] = encoder_cls

    def get(self, name: str) -> Type[ModalityEncoder]:
        if name not in self._encoders:
            raise KeyError(f"Encoder {name} not found.")
        return self._encoders[name]

    def list_encoders(self) -> list[str]:
        return list(self._encoders.keys())

    def clear(self) -> None:
        self._encoders.clear()

# ==============================================================================
# TESTS
# ==============================================================================

@pytest.fixture
def registry() -> ModalityRegistry:
    """Fixture providing a clean registry."""
    return ModalityRegistry()

def test_histopathology_encoder_shape() -> None:
    """Test HistopathologyEncoder with 4D dummy image data."""
    encoder = HistopathologyEncoder()
    data = np.random.rand(4, 256, 256, 3) # Batch, H, W, C
    encoded = encoder.encode(data)
    assert encoded.shape == (4, 3)

def test_clinical_encoder_logic() -> None:
    """Test ClinicalEncoder transformations."""
    encoder = ClinicalEncoder()
    data = np.array([[-1.0, 0.0, np.e - 1]])
    encoded = encoder.encode(data)
    np.testing.assert_allclose(encoded, [[np.log1p(1.0), 0.0, 1.0]])

def test_genomics_encoder_logic() -> None:
    """Test GenomicsEncoder thresholding."""
    encoder = GenomicsEncoder()
    data = np.array([[-0.5, 0.0, 0.5, 1.5]])
    encoded = encoder.encode(data)
    np.testing.assert_array_equal(encoded, [[0.0, 0.0, 1.0, 1.0]])

@pytest.mark.parametrize("encoder_cls", [
    HistopathologyEncoder, ClinicalEncoder, GenomicsEncoder
])
def test_encoder_is_subclass(encoder_cls: Type[ModalityEncoder]) -> None:
    """Test that all mock encoders inherit from ModalityEncoder."""
    assert issubclass(encoder_cls, ModalityEncoder)

@pytest.mark.parametrize("modality_name, encoder_cls", [
    ("histo", HistopathologyEncoder),
    ("clinical", ClinicalEncoder),
    ("genomics", GenomicsEncoder),
    ("custom", HistopathologyEncoder)
])
def test_registry_registration(
    registry: ModalityRegistry, modality_name: str, encoder_cls: Type[ModalityEncoder]
) -> None:
    """Test registering different encoders to the registry."""
    registry.register(modality_name, encoder_cls)
    assert registry.get(modality_name) is encoder_cls
    assert modality_name in registry.list_encoders()

def test_registry_overwrite_error(registry: ModalityRegistry) -> None:
    """Test that registering an existing name raises ValueError."""
    registry.register("clinical", ClinicalEncoder)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("clinical", GenomicsEncoder)

def test_registry_missing_error(registry: ModalityRegistry) -> None:
    """Test that retrieving an unknown name raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        registry.get("missing_modality")

@pytest.mark.parametrize("names", [
    [],
    ["histo"],
    ["histo", "clin"],
    ["a", "b", "c", "d", "e"]
])
def test_registry_list_encoders(registry: ModalityRegistry, names: list[str]) -> None:
    """Test listing of registered encoders."""
    for n in names:
        registry.register(n, ClinicalEncoder)
    registered = registry.list_encoders()
    assert len(registered) == len(names)
    assert set(registered) == set(names)

def test_registry_clear(registry: ModalityRegistry) -> None:
    """Test clearing the registry."""
    registry.register("a", ClinicalEncoder)
    registry.clear()
    assert len(registry.list_encoders()) == 0

@pytest.mark.parametrize("shape", [(10, 5), (32, 100), (1, 1)])
def test_clinical_encoder_preserves_shape(shape: tuple) -> None:
    """Test that clinical encoder preserves 2D shapes."""
    encoder = ClinicalEncoder()
    data = np.random.randn(*shape)
    encoded = encoder.encode(data)
    assert encoded.shape == shape

@pytest.mark.parametrize("shape", [(10, 5), (32, 100), (1, 1)])
def test_genomics_encoder_preserves_shape(shape: tuple) -> None:
    """Test that genomics encoder preserves 2D shapes."""
    encoder = GenomicsEncoder()
    data = np.random.randn(*shape)
    encoded = encoder.encode(data)
    assert encoded.shape == shape
