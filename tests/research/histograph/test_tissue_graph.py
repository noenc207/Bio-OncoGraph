import pytest
import numpy as np
from unittest.mock import MagicMock

from bio_oncograph.research.histograph.tissue_graph import (
    TissueNode,
    TissueGraph,
    TissueGraphBuilder
)


def test_tissue_node_creation() -> None:
    """Test TissueNode initialization."""
    node = TissueNode(node_id=1, region_type="tumor", center_x=50.0, center_y=50.0)
    assert node.node_id == 1
    assert node.region_type == "tumor"


def test_tissue_node_features() -> None:
    """Test TissueNode features handling."""
    feats = np.array([0.5, 0.5])
    node = TissueNode(node_id=1, region_type="stroma", center_x=0.0, center_y=0.0, features=feats)
    np.testing.assert_array_equal(node.features, feats)


def test_tissue_node_default_features() -> None:
    """Test TissueNode handles missing features."""
    node = TissueNode(node_id=1, region_type="necrosis", center_x=0.0, center_y=0.0)
    assert node.features is None


def test_tissue_node_equality() -> None:
    """Test TissueNode equality checks."""
    node1 = TissueNode(node_id=5, region_type="tumor", center_x=10.0, center_y=10.0)
    node2 = TissueNode(node_id=5, region_type="tumor", center_x=10.0, center_y=10.0)
    assert node1 == node2


def test_tissue_node_inequality() -> None:
    """Test TissueNode inequality."""
    node1 = TissueNode(node_id=5, region_type="tumor", center_x=10.0, center_y=10.0)
    node2 = TissueNode(node_id=6, region_type="tumor", center_x=10.0, center_y=10.0)
    assert node1 != node2


def test_tissue_graph_creation() -> None:
    """Test TissueGraph initialization."""
    graph = TissueGraph(nodes=[], edge_index=np.empty((2, 0), dtype=np.int64))
    assert len(graph.nodes) == 0


def test_tissue_graph_get_node() -> None:
    """Test retrieving nodes from TissueGraph."""
    node = TissueNode(node_id=42, region_type="lymphocyte", center_x=1.0, center_y=1.0)
    graph = TissueGraph(nodes=[node], edge_index=np.empty((2, 0)))
    assert graph.get_node(42) == node


def test_tissue_graph_missing_node() -> None:
    """Test retrieving missing node raises error."""
    graph = TissueGraph(nodes=[], edge_index=np.empty((2, 0)))
    with pytest.raises(KeyError):
        graph.get_node(100)


def test_tissue_graph_builder_init() -> None:
    """Test TissueGraphBuilder initialization."""
    builder = TissueGraphBuilder()
    assert builder is not None


def test_tissue_graph_builder_add_node() -> None:
    """Test adding node to TissueGraphBuilder."""
    builder = TissueGraphBuilder()
    builder.add_node(region_type="tumor", center_x=15.0, center_y=25.0)
    assert builder.node_count == 1


def test_tissue_graph_builder_build_empty() -> None:
    """Test building empty TissueGraph."""
    builder = TissueGraphBuilder()
    graph = builder.build()
    assert len(graph.nodes) == 0


def test_tissue_graph_builder_build_knn() -> None:
    """Test building TissueGraph with KNN."""
    builder = TissueGraphBuilder(strategy="knn", knn_k=1)
    builder.add_node(region_type="t", center_x=0.0, center_y=0.0)
    builder.add_node(region_type="s", center_x=1.0, center_y=1.0)
    graph = builder.build()
    assert len(graph.nodes) == 2
    assert graph.edge_index.shape == (2, 2)


def test_tissue_graph_builder_build_radius() -> None:
    """Test building TissueGraph with radius strategy."""
    builder = TissueGraphBuilder(strategy="radius", radius=2.0)
    builder.add_node(region_type="t", center_x=0.0, center_y=0.0)
    builder.add_node(region_type="t", center_x=1.0, center_y=0.0)
    graph = builder.build()
    assert graph.edge_index.shape[1] > 0


def test_tissue_graph_builder_invalid_strategy() -> None:
    """Test builder raises error on invalid strategy."""
    builder = TissueGraphBuilder(strategy="magic")
    builder.add_node(region_type="t", center_x=0.0, center_y=0.0)
    with pytest.raises(ValueError):
        builder.build()


def test_tissue_graph_feature_matrix() -> None:
    """Test extracting feature matrix from TissueGraph."""
    builder = TissueGraphBuilder()
    builder.add_node(region_type="a", center_x=0.0, center_y=0.0, features=np.array([1.0, 2.0]))
    builder.add_node(region_type="b", center_x=1.0, center_y=1.0, features=np.array([3.0, 4.0]))
    graph = builder.build()
    mat = graph.get_feature_matrix()
    assert mat.shape == (2, 2)
    assert mat[1, 0] == 3.0


def test_tissue_graph_edge_attributes() -> None:
    """Test TissueGraph handles edge attributes."""
    nodes = [TissueNode(node_id=0, region_type="a", center_x=0.0, center_y=0.0)]
    edges = np.empty((2, 0), dtype=np.int64)
    attrs = np.empty((0, 3))
    graph = TissueGraph(nodes=nodes, edge_index=edges, edge_attr=attrs)
    assert graph.edge_attr.shape == (0, 3)
