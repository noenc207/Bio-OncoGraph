import numpy as np
import pytest
from bio_oncograph.ai.model.fusion import FeatureFusion

def test_fusion_init() -> None:
    """Test FeatureFusion initialization."""
    fusion = FeatureFusion(graph_dim=64, aux_dim=32, output_dim=128)
    assert fusion.graph_dim == 64
    assert fusion.aux_dim == 32
    assert fusion.output_dim == 128

def test_fusion_forward_shape() -> None:
    """Test output shape of fusion forward pass."""
    fusion = FeatureFusion(graph_dim=64, aux_dim=32, output_dim=128)
    graph_embed = np.random.rand(4, 64)
    aux_features = np.random.rand(4, 32)
    
    output = fusion.forward(graph_embed, aux_features)
    assert output.shape == (4, 128)

def test_fusion_concatenation() -> None:
    """Test that fusion internally combines features (mock test)."""
    # Assuming the fusion module has a straightforward concatenation 
    # before projecting to output_dim. We test the API here.
    fusion = FeatureFusion(graph_dim=10, aux_dim=5, output_dim=15)
    graph_embed = np.ones((2, 10))
    aux_features = np.zeros((2, 5))
    
    output = fusion.forward(graph_embed, aux_features)
    assert output.shape == (2, 15)

def test_fusion_missing_aux() -> None:
    """Test fusion when auxiliary features might be missing or empty."""
    fusion = FeatureFusion(graph_dim=64, aux_dim=0, output_dim=128)
    graph_embed = np.random.rand(4, 64)
    aux_features = np.empty((4, 0))
    
    output = fusion.forward(graph_embed, aux_features)
    assert output.shape == (4, 128)

def test_fusion_dimension_mismatch() -> None:
    """Test fusion raises error on dimension mismatch."""
    fusion = FeatureFusion(graph_dim=64, aux_dim=32, output_dim=128)
    graph_embed = np.random.rand(4, 64)
    aux_features = np.random.rand(4, 16)  # Wrong aux_dim
    
    with pytest.raises((ValueError, RuntimeError, Exception)):
        fusion.forward(graph_embed, aux_features)

def test_fusion_deterministic() -> None:
    """Test fusion produces deterministic output for same input."""
    fusion = FeatureFusion(graph_dim=16, aux_dim=16, output_dim=32)
    graph_embed = np.ones((1, 16))
    aux_features = np.zeros((1, 16))
    
    out1 = fusion.forward(graph_embed, aux_features)
    out2 = fusion.forward(graph_embed, aux_features)
    
    np.testing.assert_array_almost_equal(out1, out2)

def test_fusion_batch_size_mismatch() -> None:
    """Test fusion raises error when batch sizes mismatch."""
    fusion = FeatureFusion(graph_dim=16, aux_dim=16, output_dim=32)
    graph_embed = np.random.rand(4, 16)
    aux_features = np.random.rand(5, 16)
    
    with pytest.raises((ValueError, RuntimeError, Exception)):
        fusion.forward(graph_embed, aux_features)
