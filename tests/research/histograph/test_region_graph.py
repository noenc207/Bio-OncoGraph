import pytest
import numpy as np

from bio_oncograph.research.histograph.region_graph import (
    RegionNode,
    RegionGraph,
    RegionGraphBuilder
)


def test_region_node_init() -> None:
    """Test RegionNode initialization."""
    node = RegionNode(node_id=1, label="stroma_rich", area=150.5, centroid=(10.0, 20.0))
    assert node.node_id == 1
    assert node.label == "stroma_rich"
    assert node.area == 150.5


def test_region_node_with_features() -> None:
    """Test RegionNode with features array."""
    feats = np.ones(10)
    node = RegionNode(node_id=2, label="tumor_core", area=500.0, centroid=(0., 0.), features=feats)
    assert node.features.shape == (10,)
    assert node.features[0] == 1.0


def test_region_node_equality() -> None:
    """Test RegionNode equality."""
    n1 = RegionNode(node_id=1, label="a", area=10.0, centroid=(0., 0.))
    n2 = RegionNode(node_id=1, label="a", area=10.0, centroid=(0., 0.))
    assert n1 == n2


def test_region_node_inequality() -> None:
    """Test RegionNode inequality."""
    n1 = RegionNode(node_id=1, label="a", area=10.0, centroid=(0., 0.))
    n2 = RegionNode(node_id=2, label="b", area=10.0, centroid=(0., 0.))
    assert n1 != n2


def test_region_graph_init_empty() -> None:
    """Test RegionGraph initialization with no nodes."""
    graph = RegionGraph(nodes=[], edge_index=np.empty((2, 0), dtype=np.int64))
    assert len(graph.nodes) == 0


def test_region_graph_get_node() -> None:
    """Test RegionGraph get_node method."""
    n = RegionNode(node_id=99, label="necrosis", area=50.0, centroid=(5., 5.))
    graph = RegionGraph(nodes=[n], edge_index=np.empty((2, 0)))
    assert graph.get_node(99) == n


def test_region_graph_get_node_missing() -> None:
    """Test RegionGraph raises KeyError for missing node."""
    graph = RegionGraph(nodes=[], edge_index=np.empty((2, 0)))
    with pytest.raises(KeyError):
        graph.get_node(1)


def test_region_graph_builder_init() -> None:
    """Test RegionGraphBuilder initialization."""
    builder = RegionGraphBuilder()
    assert builder is not None


def test_region_graph_builder_add_node() -> None:
    """Test adding node to RegionGraphBuilder."""
    builder = RegionGraphBuilder()
    builder.add_node(label="x", area=10.0, centroid=(0., 0.))
    assert builder.node_count == 1


def test_region_graph_builder_build_empty() -> None:
    """Test RegionGraphBuilder building empty graph."""
    builder = RegionGraphBuilder()
    graph = builder.build()
    assert len(graph.nodes) == 0


def test_region_graph_builder_build_fully_connected() -> None:
    """Test RegionGraphBuilder with fully connected strategy."""
    builder = RegionGraphBuilder(strategy="fully_connected")
    builder.add_node(label="x", area=10.0, centroid=(0., 0.))
    builder.add_node(label="y", area=20.0, centroid=(1., 1.))
    graph = builder.build()
    assert graph.edge_index.shape == (2, 2)


def test_region_graph_builder_build_knn() -> None:
    """Test RegionGraphBuilder with KNN strategy."""
    builder = RegionGraphBuilder(strategy="knn", knn_k=1)
    builder.add_node(label="a", area=10.0, centroid=(0., 0.))
    builder.add_node(label="b", area=10.0, centroid=(0.1, 0.1))
    graph = builder.build()
    assert graph.edge_index.shape[1] == 2


def test_region_graph_builder_reset() -> None:
    """Test RegionGraphBuilder reset."""
    builder = RegionGraphBuilder()
    builder.add_node(label="a", area=10.0, centroid=(0., 0.))
    builder.reset()
    assert builder.node_count == 0
