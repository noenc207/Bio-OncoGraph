"""Tests for embedding containers and metadata."""
import pytest
import numpy as np

class EmbeddingMetadata:
    """Metadata for embeddings."""
    def __init__(self, model_name: str, dimension: int):
        self._model_name = model_name
        self._dimension = dimension
        
    @property
    def model_name(self) -> str:
        return self._model_name
        
    @property
    def dimension(self) -> int:
        return self._dimension

class EmbeddingContainer:
    """Container holding data and metadata."""
    def __init__(self, data: np.ndarray, metadata: EmbeddingMetadata):
        self._data = data
        self._metadata = metadata
        
    @property
    def data(self) -> np.ndarray:
        return self._data
        
    @property
    def metadata(self) -> EmbeddingMetadata:
        return self._metadata


@pytest.mark.parametrize("model_name", ["model_A", "model_B", "model_C", "bert", "vit"])
@pytest.mark.parametrize("dimension", [64, 128, 256, 512, 768, 1024])
@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_embedding_metadata_and_container(model_name: str, dimension: int, batch_size: int) -> None:
    """Test embedding metadata and container hold shape and remain immutable."""
    meta = EmbeddingMetadata(model_name=model_name, dimension=dimension)
    assert meta.model_name == model_name
    assert meta.dimension == dimension
    
    data = np.random.rand(batch_size, dimension)
    container = EmbeddingContainer(data=data, metadata=meta)
    
    assert container.data.shape == (batch_size, dimension)
    assert container.metadata is meta
    
    with pytest.raises(AttributeError):
        container.data = np.random.rand(batch_size, dimension)
    with pytest.raises(AttributeError):
        container.metadata = meta

@pytest.mark.parametrize("model_name", ["vit-base", "resnet50", "densenet"])
@pytest.mark.parametrize("dimension", [768, 2048, 1024])
@pytest.mark.parametrize("repeat", list(range(5)))
def test_embedding_metadata_immutability(model_name: str, dimension: int, repeat: int) -> None:
    """Test that embedding metadata attributes cannot be modified."""
    meta = EmbeddingMetadata(model_name=model_name, dimension=dimension)
    with pytest.raises(AttributeError):
        meta.model_name = "new_model"
    with pytest.raises(AttributeError):
        meta.dimension = 128
