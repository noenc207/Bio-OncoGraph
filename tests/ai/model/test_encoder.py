import numpy as np
import pytest
from bio_oncograph.ai.model.encoder import GraphEncoder

def test_encoder_init() -> None:
    """Test initialization of GraphEncoder."""
    encoder = GraphEncoder(input_dim=128, hidden_dim=64, num_layers=3)
    assert encoder.input_dim == 128
    assert encoder.hidden_dim == 64
    assert encoder.num_layers == 3

def test_encoder_forward_shape() -> None:
    """Test the output shape of the forward pass."""
    encoder = GraphEncoder(input_dim=128, hidden_dim=64, num_layers=3)
    node_features = np.random.rand(10, 128)
    edge_index = np.random.randint(0, 10, size=(2, 20))
    output = encoder.forward(node_features, edge_index)
    
    assert isinstance(output, dict)
    assert "node_embeddings" in output
    assert output["node_embeddings"].shape == (10, 64)

def test_encoder_deterministic_output() -> None:
    """Test that the encoder produces deterministic outputs."""
    encoder = GraphEncoder(input_dim=64, hidden_dim=32, num_layers=2)
    node_features = np.ones((5, 64))
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    
    out1 = encoder.forward(node_features, edge_index)
    out2 = encoder.forward(node_features, edge_index)
    
    np.testing.assert_array_almost_equal(out1["node_embeddings"], out2["node_embeddings"])

def test_encoder_invalid_input_shape() -> None:
    """Test encoder behavior with invalid input shapes."""
    encoder = GraphEncoder(input_dim=128, hidden_dim=64, num_layers=2)
    node_features = np.random.rand(10, 64)  # Wrong input_dim
    edge_index = np.random.randint(0, 10, size=(2, 20))
    
    with pytest.raises((ValueError, RuntimeError, Exception)):
        encoder.forward(node_features, edge_index)

def test_encoder_embedding_dim() -> None:
    """Test that embedding dimensions match config."""
    hidden_dim = 16
    encoder = GraphEncoder(input_dim=32, hidden_dim=hidden_dim, num_layers=1)
    assert encoder.hidden_dim == hidden_dim

def test_encoder_multiple_nodes() -> None:
    """Test encoder with a large number of nodes."""
    encoder = GraphEncoder(input_dim=16, hidden_dim=8, num_layers=2)
    node_features = np.random.rand(1000, 16)
    edge_index = np.random.randint(0, 1000, size=(2, 5000))
    
    output = encoder.forward(node_features, edge_index)
    assert output["node_embeddings"].shape == (1000, 8)

def test_encoder_no_edges() -> None:
    """Test encoder with graph containing no edges."""
    encoder = GraphEncoder(input_dim=32, hidden_dim=16, num_layers=2)
    node_features = np.random.rand(5, 32)
    edge_index = np.empty((2, 0), dtype=int)
    
    output = encoder.forward(node_features, edge_index)
    assert output["node_embeddings"].shape == (5, 16)
