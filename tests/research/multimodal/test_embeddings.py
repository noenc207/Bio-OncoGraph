import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ==============================================================================
# MOCKS
# ==============================================================================
@dataclass
class ModalityEmbedding:
    modality_name: str
    data: np.ndarray

@dataclass
class FusedEmbedding:
    fusion_type: str
    data: np.ndarray
    source_modalities: list[str]

class EmbeddingStore:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def save(self, key: str, embedding: Any) -> None:
        self._store[key] = embedding

    def load(self, key: str) -> Any:
        if key not in self._store:
            raise KeyError(f"Embedding {key} not found.")
        return self._store[key]
    
    def clear(self) -> None:
        self._store.clear()

    def get_all_keys(self) -> list[str]:
        return list(self._store.keys())

# ==============================================================================
# TESTS
# ==============================================================================

@pytest.fixture
def store() -> EmbeddingStore:
    return EmbeddingStore()

def test_modality_embedding_creation() -> None:
    """Test creation of ModalityEmbedding."""
    data = np.random.rand(10, 10)
    emb = ModalityEmbedding(modality_name="clinical", data=data)
    assert emb.modality_name == "clinical"
    np.testing.assert_array_equal(emb.data, data)

def test_fused_embedding_creation() -> None:
    """Test creation of FusedEmbedding."""
    data = np.random.rand(10, 20)
    emb = FusedEmbedding(fusion_type="early", data=data, source_modalities=["cli", "gen"])
    assert emb.fusion_type == "early"
    assert emb.source_modalities == ["cli", "gen"]
    np.testing.assert_array_equal(emb.data, data)

@pytest.mark.parametrize("key", ["test1", "patient_001", "uid-999"])
def test_embedding_store_save_load(store: EmbeddingStore, key: str) -> None:
    """Test save and load functionality of EmbeddingStore."""
    emb = ModalityEmbedding("clinical", np.array([1, 2, 3]))
    store.save(key, emb)
    loaded = store.load(key)
    assert loaded.modality_name == "clinical"
    np.testing.assert_array_equal(loaded.data, emb.data)

def test_embedding_store_missing_key(store: EmbeddingStore) -> None:
    """Test missing key raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        store.load("nonexistent_key")

def test_embedding_store_clear(store: EmbeddingStore) -> None:
    """Test clearing the store."""
    store.save("k1", ModalityEmbedding("m1", np.zeros(2)))
    store.save("k2", ModalityEmbedding("m2", np.zeros(2)))
    store.clear()
    assert len(store.get_all_keys()) == 0

@pytest.mark.parametrize("count", [0, 1, 5, 10])
def test_embedding_store_get_all_keys(store: EmbeddingStore, count: int) -> None:
    """Test retrieval of all keys."""
    keys = [f"k_{i}" for i in range(count)]
    for k in keys:
        store.save(k, ModalityEmbedding("m", np.zeros(1)))
    assert set(store.get_all_keys()) == set(keys)
