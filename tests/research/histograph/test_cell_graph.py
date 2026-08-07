import pytest
import numpy as np
from unittest.mock import MagicMock

from bio_oncograph.research.histograph.cell_graph import (
    CellNode,
    CellGraph,
    CellGraphBuilder
)


def test_cell_node_creation() -> None:
    """Test CellNode creation with standard inputs."""
    node = CellNode(node_id=1, x=10.0, y=20.0, features=np.array([0.1, 0.2]))
    assert node.node_id == 1
    assert node.x == 10.0


def test_cell_node_features_shape() -> None:
    """Test CellNode feature array shape retention."""
    feats = np.random.rand(64)
    node = CellNode(node_id=2, x=0.0, y=0.0, features=feats)
    assert node.features.shape == (64,)


def test_cell_node_equality() -> None:
    """Test CellNode equality based on id."""
    node1 = CellNode(node_id=1, x=10.0, y=20.0, features=np.array([0.1]))
    node2 = CellNode(node_id=1, x=10.0, y=20.0, features=np.array([0.1]))
    assert node1 == node2


def test_cell_node_inequality() -> None:
    """Test CellNode inequality."""
    node1 = CellNode(node_id=1, x=10.0, y=20.0, features=np.array([0.1]))
    node2 = CellNode(node_id=2, x=10.0, y=20.0, features=np.array([0.1]))
    assert node1 != node2


def test_cell_graph_creation_empty() -> None:
    """Test CellGraph creation with empty nodes."""
    graph = CellGraph(nodes=[], edge_index=np.empty((2, 0), dtype=np.int64))
    assert len(graph.nodes) == 0


def test_cell_graph_creation_populated() -> None:
    """Test CellGraph creation with nodes and edges."""
    node = CellNode(node_id=0, x=0.0, y=0.0, features=np.array([1.0]))
    edges = np.array([[0], [0]], dtype=np.int64)
    graph = CellGraph(nodes=[node], edge_index=edges)
    assert len(graph.nodes) == 1
    assert graph.edge_index.shape == (2, 1)


def test_cell_graph_node_retrieval() -> None:
    """Test retrieving a node from CellGraph."""
    node = CellNode(node_id=0, x=0.0, y=0.0, features=np.array([1.0]))
    graph = CellGraph(nodes=[node], edge_index=np.empty((2, 0)))
    assert graph.get_node(0) == node


def test_cell_graph_invalid_node_retrieval() -> None:
    """Test retrieving non-existent node raises error."""
    graph = CellGraph(nodes=[], edge_index=np.empty((2, 0)))
    with pytest.raises(KeyError):
        graph.get_node(99)


def test_cell_graph_builder_initialization() -> None:
    """Test CellGraphBuilder initialization."""
    builder = CellGraphBuilder()
    assert builder is not None


def test_cell_graph_builder_add_node() -> None:
    """Test adding a single node to builder."""
    builder = CellGraphBuilder()
    builder.add_node(x=1.0, y=2.0, features=np.array([0.5]))
    assert builder.node_count == 1


def test_cell_graph_builder_add_multiple_nodes() -> None:
    """Test adding multiple nodes to builder."""
    builder = CellGraphBuilder()
    for i in range(5):
        builder.add_node(x=float(i), y=float(i), features=np.array([0.1]))
    assert builder.node_count == 5


def test_cell_graph_builder_build_empty() -> None:
    """Test building a graph with zero nodes."""
    builder = CellGraphBuilder()
    graph = builder.build()
    assert len(graph.nodes) == 0
    assert graph.edge_index.shape == (2, 0)


def test_cell_graph_builder_build_knn() -> None:
    """Test building a graph using KNN strategy."""
    builder = CellGraphBuilder(strategy="knn", knn_k=2)
    for i in range(4):
        builder.add_node(x=float(i), y=float(i), features=np.array([0.1]))
    graph = builder.build()
    assert len(graph.nodes) == 4
    assert graph.edge_index.shape[0] == 2


def test_cell_graph_builder_build_radius() -> None:
    """Test building a graph using radius strategy."""
    builder = CellGraphBuilder(strategy="radius", radius=1.5)
    builder.add_node(x=0.0, y=0.0, features=np.array([1.0]))
    builder.add_node(x=1.0, y=0.0, features=np.array([1.0]))
    builder.add_node(x=10.0, y=10.0, features=np.array([1.0]))
    graph = builder.build()
    assert len(graph.nodes) == 3


def test_cell_graph_builder_determinism() -> None:
    """Test that graph building is deterministic."""
    coords = [(0., 0.), (1., 1.), (2., 2.)]
    b1 = CellGraphBuilder(strategy="fully_connected")
    b2 = CellGraphBuilder(strategy="fully_connected")
    for x, y in coords:
        b1.add_node(x=x, y=y, features=np.array([0.0]))
        b2.add_node(x=x, y=y, features=np.array([0.0]))
    g1 = b1.build()
    g2 = b2.build()
    np.testing.assert_array_equal(g1.edge_index, g2.edge_index)


def test_cell_graph_builder_invalid_strategy() -> None:
    """Test invalid strategy raises error on build."""
    builder = CellGraphBuilder(strategy="invalid_strat")
    builder.add_node(x=0.0, y=0.0, features=np.array([0.]))
    with pytest.raises(ValueError):
        builder.build()


def test_cell_graph_node_features_matrix() -> None:
    """Test generation of node features matrix."""
    builder = CellGraphBuilder()
    builder.add_node(x=0.0, y=0.0, features=np.array([0.1, 0.2]))
    builder.add_node(x=1.0, y=1.0, features=np.array([0.3, 0.4]))
    graph = builder.build()
    feat_mat = graph.get_feature_matrix()
    assert feat_mat.shape == (2, 2)
    np.testing.assert_array_equal(feat_mat[1], np.array([0.3, 0.4]))


def test_cell_graph_edge_weights_default() -> None:
    """Test that edge weights are None by default."""
    builder = CellGraphBuilder()
    builder.add_node(x=0.0, y=0.0, features=np.array([0.]))
    graph = builder.build()
    assert graph.edge_weights is None


def test_cell_graph_builder_reset() -> None:
    """Test builder reset functionality."""
    builder = CellGraphBuilder()
    builder.add_node(x=0.0, y=0.0, features=np.array([0.]))
    builder.reset()
    assert builder.node_count == 0
