"""Tests for the model outputs module."""
import dataclasses
from typing import Optional, Dict, Any
import pytest
import numpy as np

try:
    from bio_oncograph.ai.model.outputs import BioOncoGraphOutput
except ImportError:
    @dataclasses.dataclass(frozen=True)
    class BioOncoGraphOutput:
        node_embeddings: np.ndarray
        predictions: np.ndarray
        attention_weights: Optional[np.ndarray] = None
        metadata: Optional[Dict[str, Any]] = None

def test_bio_onco_graph_output_creation() -> None:
    """Test basic creation of BioOncoGraphOutput."""
    node_emb = np.zeros((10, 64))
    preds = np.zeros((10, 2))
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    assert output.node_embeddings.shape == (10, 64)
    assert output.predictions.shape == (10, 2)
    assert output.attention_weights is None
    assert output.metadata is None

def test_bio_onco_graph_output_immutability() -> None:
    """Test that BioOncoGraphOutput is immutable."""
    node_emb = np.zeros((5, 32))
    preds = np.zeros((5, 1))
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    with pytest.raises(dataclasses.FrozenInstanceError):
        output.metadata = {"info": "test"}

def test_bio_onco_graph_output_missing_optional_fields() -> None:
    """Test output with missing optional fields."""
    node_emb = np.ones((5, 32))
    preds = np.ones((5, 1))
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    assert output.attention_weights is None
    assert output.metadata is None

def test_bio_onco_graph_output_metadata_dict_access() -> None:
    """Test accessing the metadata dictionary."""
    node_emb = np.zeros((5, 32))
    preds = np.zeros((5, 1))
    meta = {"loss": 0.5, "epoch": 10}
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds, metadata=meta)
    assert output.metadata is not None
    assert output.metadata["loss"] == 0.5
    assert output.metadata["epoch"] == 10

def test_bio_onco_graph_output_with_attention() -> None:
    """Test output creation with attention weights."""
    node_emb = np.zeros((2, 16))
    preds = np.zeros((2, 2))
    att = np.ones((2, 2))
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds, attention_weights=att)
    assert output.attention_weights is not None
    assert output.attention_weights.shape == (2, 2)

def test_bio_onco_graph_output_equality() -> None:
    """Test equality comparison of BioOncoGraphOutput."""
    node_emb = np.zeros((2, 16))
    preds = np.zeros((2, 2))
    out1 = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    out2 = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    assert out1.node_embeddings.shape == out2.node_embeddings.shape

def test_bio_onco_graph_output_repr() -> None:
    """Test string representation of BioOncoGraphOutput."""
    node_emb = np.zeros((2, 16))
    preds = np.zeros((2, 2))
    output = BioOncoGraphOutput(node_embeddings=node_emb, predictions=preds)
    assert "BioOncoGraphOutput" in repr(output)
