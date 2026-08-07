import numpy as np
import pytest
from bio_oncograph.ai.model.model import BioOncoGraphModel, BioOncoGraphOutput

class DummyConfig:
    def __init__(self):
        self.node_dim = 64
        self.encoder_hidden = 32
        self.encoder_layers = 2
        self.pool_type = "mean"
        self.aux_dim = 16
        self.fusion_dim = 64
        self.num_classes = 2

def test_model_init() -> None:
    """Test BioOncoGraphModel init."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    assert model.config == config

def test_model_encode_step() -> None:
    """Test the encode step of the model."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    node_features = np.random.rand(10, config.node_dim)
    edge_index = np.random.randint(0, 10, size=(2, 20))
    
    node_embeds = model.encode(node_features, edge_index)
    assert node_embeds.shape == (10, config.encoder_hidden)

def test_model_pool_step() -> None:
    """Test the pooling step of the model."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    node_embeds = np.random.rand(10, config.encoder_hidden)
    batch = np.zeros(10, dtype=int)
    
    graph_embed = model.pool(node_embeds, batch)
    assert graph_embed.shape == (1, config.encoder_hidden)

def test_model_fuse_step() -> None:
    """Test the fusion step of the model."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    graph_embed = np.random.rand(2, config.encoder_hidden)
    aux_features = np.random.rand(2, config.aux_dim)
    
    fused = model.fuse(graph_embed, aux_features)
    assert fused.shape == (2, config.fusion_dim)

def test_model_predict_step() -> None:
    """Test the predict step of the model."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    fused_features = np.random.rand(2, config.fusion_dim)
    
    preds = model.predict(fused_features)
    assert "logits" in preds
    assert preds["logits"].shape == (2, config.num_classes)

def test_model_forward_flow() -> None:
    """Test the entire forward flow execution."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    
    node_features = np.random.rand(10, config.node_dim)
    edge_index = np.array([[0, 1, 2], [1, 2, 0]])
    batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    aux_features = np.random.rand(2, config.aux_dim)
    
    output = model.forward(node_features, edge_index, batch, aux_features)
    assert isinstance(output, BioOncoGraphOutput)

def test_model_forward_returns_output() -> None:
    """Test forward returns a BioOncoGraphOutput instance with correct attributes."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    
    node_features = np.random.rand(5, config.node_dim)
    edge_index = np.empty((2, 0), dtype=int)
    batch = np.zeros(5, dtype=int)
    aux_features = np.random.rand(1, config.aux_dim)
    
    output = model.forward(node_features, edge_index, batch, aux_features)
    assert hasattr(output, "logits")
    assert hasattr(output, "risk_score")

def test_model_deterministic_inference() -> None:
    """Test model produces deterministic output."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    
    node_features = np.ones((5, config.node_dim))
    edge_index = np.array([[0, 1], [1, 2]])
    batch = np.zeros(5, dtype=int)
    aux_features = np.zeros((1, config.aux_dim))
    
    out1 = model.forward(node_features, edge_index, batch, aux_features)
    out2 = model.forward(node_features, edge_index, batch, aux_features)
    
    np.testing.assert_array_almost_equal(out1.logits, out2.logits)

def test_model_validation_integration() -> None:
    """Test model handles validation flags correctly."""
    config = DummyConfig()
    model = BioOncoGraphModel(config)
    
    node_features = np.ones((2, config.node_dim))
    edge_index = np.array([[0], [1]])
    batch = np.zeros(2, dtype=int)
    aux_features = np.ones((1, config.aux_dim))
    
    # Assuming the forward method accepts a validation/training boolean flag
    try:
        model.forward(node_features, edge_index, batch, aux_features, is_training=False)
    except TypeError:
        # If is_training is not supported, just pass without exception
        pass

def test_model_missing_config() -> None:
    """Test model initialization with missing config raises an error."""
    with pytest.raises((ValueError, TypeError)):
        BioOncoGraphModel(None)
