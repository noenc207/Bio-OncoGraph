import numpy as np
import pytest
from bio_oncograph.ai.model.pooling import GraphPooling

def test_pooling_init() -> None:
    """Test GraphPooling initialization."""
    pooler = GraphPooling(pool_type="mean")
    assert pooler.pool_type == "mean"

def test_pooling_max_pooling() -> None:
    """Test max pooling logic."""
    pooler = GraphPooling(pool_type="max")
    node_embeddings = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 1.0, 0.0],
        [2.0, 5.0, 1.0]
    ])
    # batch mapping: node 0->graph 0, node 1->graph 0, node 2->graph 1
    batch = np.array([0, 0, 1])
    output = pooler.forward(node_embeddings, batch)
    
    assert output.shape == (2, 3)
    np.testing.assert_array_almost_equal(output[0], [4.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(output[1], [2.0, 5.0, 1.0])

def test_pooling_mean_pooling() -> None:
    """Test mean pooling logic."""
    pooler = GraphPooling(pool_type="mean")
    node_embeddings = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
        [2.0, 2.0, 2.0]
    ])
    batch = np.array([0, 0, 1])
    output = pooler.forward(node_embeddings, batch)
    
    assert output.shape == (2, 3)
    np.testing.assert_array_almost_equal(output[0], [2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(output[1], [2.0, 2.0, 2.0])

def test_pooling_sum_pooling() -> None:
    """Test sum pooling logic."""
    pooler = GraphPooling(pool_type="sum")
    node_embeddings = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    batch = np.array([0, 0])
    output = pooler.forward(node_embeddings, batch)
    
    assert output.shape == (1, 2)
    np.testing.assert_array_almost_equal(output[0], [4.0, 6.0])

def test_pooling_invalid_type() -> None:
    """Test pooling with unsupported type."""
    with pytest.raises(ValueError):
        GraphPooling(pool_type="invalid_type")

def test_pooling_empty_graph() -> None:
    """Test pooling with no nodes."""
    pooler = GraphPooling(pool_type="mean")
    node_embeddings = np.empty((0, 16))
    batch = np.empty((0,), dtype=int)
    
    output = pooler.forward(node_embeddings, batch)
    assert output.shape == (0, 16)

def test_pooling_output_shape() -> None:
    """Test output shape when multiple graphs are in the batch."""
    pooler = GraphPooling(pool_type="mean")
    node_embeddings = np.random.rand(10, 64)
    batch = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    
    output = pooler.forward(node_embeddings, batch)
    assert output.shape == (4, 64)
