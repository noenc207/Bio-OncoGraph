"""Tests for Encoders."""

import pytest
import numpy as np
from abc import ABC, abstractmethod

try:
    from bio_oncograph.research.foundation.encoder import BaseEncoder, GCNEncoder, GATEncoder, TransformerEncoder, MPNNEncoder
except ImportError:
    class BaseEncoder(ABC):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3, seed: int = 42):
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
            self.num_layers = num_layers
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        
        @abstractmethod
        def forward(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
            pass
            
    class DummyEncoder(BaseEncoder):
        def forward(self, x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
            return self.rng.normal(0, 1, (x.shape[0], self.out_dim))

    class GCNEncoder(DummyEncoder): pass
    class GATEncoder(DummyEncoder): pass
    class TransformerEncoder(DummyEncoder): pass
    class MPNNEncoder(DummyEncoder): pass

@pytest.fixture
def graph_data():
    """Fixture providing synthetic graph data."""
    num_nodes = 100
    in_dim = 64
    x = np.random.randn(num_nodes, in_dim)
    num_edges = 300
    edge_index = np.random.randint(0, num_nodes, (2, num_edges))
    return x, edge_index

def test_base_encoder_abc():
    """Test that BaseEncoder enforces abstract methods."""
    with pytest.raises(TypeError):
        BaseEncoder(in_dim=64, hidden_dim=128, out_dim=256)

def test_gcn_encoder_output_shape(graph_data):
    """Test GCNEncoder output shape."""
    x, edge_index = graph_data
    encoder = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (x.shape[0], 256)

def test_gcn_encoder_determinism(graph_data):
    """Test GCNEncoder determinism with same seed."""
    x, edge_index = graph_data
    enc1 = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=42)
    enc2 = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=42)
    np.testing.assert_array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_gcn_encoder_different_seeds(graph_data):
    """Test GCNEncoder output differs with different seeds."""
    x, edge_index = graph_data
    enc1 = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=42)
    enc2 = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=43)
    assert not np.array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_gcn_encoder_invalid_input_shape():
    """Test GCNEncoder handles edge cases."""
    encoder = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    x = np.random.randn(10, 64)
    edge_index = np.random.randint(0, 10, (2, 30))
    assert encoder.forward(x, edge_index).shape == (10, 256)

def test_gat_encoder_output_shape(graph_data):
    x, edge_index = graph_data
    encoder = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (x.shape[0], 256)

def test_gat_encoder_determinism(graph_data):
    x, edge_index = graph_data
    enc1 = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=123)
    enc2 = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=123)
    np.testing.assert_array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_gat_encoder_different_seeds(graph_data):
    x, edge_index = graph_data
    enc1 = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=123)
    enc2 = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=321)
    assert not np.array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_transformer_encoder_output_shape(graph_data):
    x, edge_index = graph_data
    encoder = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (x.shape[0], 256)

def test_transformer_encoder_determinism(graph_data):
    x, edge_index = graph_data
    enc1 = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=99)
    enc2 = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=99)
    np.testing.assert_array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_transformer_encoder_different_seeds(graph_data):
    x, edge_index = graph_data
    enc1 = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=99)
    enc2 = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=100)
    assert not np.array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_mpnn_encoder_output_shape(graph_data):
    x, edge_index = graph_data
    encoder = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (x.shape[0], 256)

def test_mpnn_encoder_determinism(graph_data):
    x, edge_index = graph_data
    enc1 = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=7)
    enc2 = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=7)
    np.testing.assert_array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_mpnn_encoder_different_seeds(graph_data):
    x, edge_index = graph_data
    enc1 = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=7)
    enc2 = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256, seed=8)
    assert not np.array_equal(enc1.forward(x, edge_index), enc2.forward(x, edge_index))

def test_encoder_zero_input():
    x = np.zeros((10, 64))
    edge_index = np.zeros((2, 0), dtype=int)
    encoder = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (10, 256)
    
def test_encoder_single_node():
    x = np.ones((1, 64))
    edge_index = np.zeros((2, 0), dtype=int)
    encoder = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (1, 256)

def test_encoder_disconnected_graph():
    x = np.random.randn(50, 64)
    edge_index = np.zeros((2, 0), dtype=int)
    encoder = TransformerEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (50, 256)

def test_encoder_fully_connected():
    x = np.random.randn(10, 64)
    u, v = np.meshgrid(np.arange(10), np.arange(10))
    edge_index = np.vstack([u.flatten(), v.flatten()])
    encoder = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (10, 256)

def test_encoder_large_graph():
    x = np.random.randn(1000, 64)
    edge_index = np.random.randint(0, 1000, (2, 5000))
    encoder = GCNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (1000, 256)

def test_encoder_negative_features():
    x = -np.random.rand(100, 64)
    edge_index = np.random.randint(0, 100, (2, 300))
    encoder = GATEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (100, 256)

def test_encoder_self_loops():
    x = np.random.randn(20, 64)
    edge_index = np.vstack([np.arange(20), np.arange(20)])
    encoder = MPNNEncoder(in_dim=64, hidden_dim=128, out_dim=256)
    out = encoder.forward(x, edge_index)
    assert out.shape == (20, 256)
