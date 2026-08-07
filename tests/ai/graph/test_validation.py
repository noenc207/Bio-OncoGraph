"""Tests for graph validation functions."""

import pytest
import warnings

from bio_oncograph.ai.graph.node import Node
from bio_oncograph.ai.graph.edge import Edge
from bio_oncograph.ai.graph.graph import Graph
from bio_oncograph.ai.graph.validation import GraphConfig, validate_graph


@pytest.fixture
def base_config() -> GraphConfig:
    """Provide a basic graph configuration for tests."""
    return GraphConfig(
        directed=True,
        weighted=True,
        allow_self_loops=False,
        allow_parallel_edges=False
    )


def test_validate_valid_graph(base_config: GraphConfig) -> None:
    """Test that a valid graph passes validation without errors."""
    nodes = (Node(id="n1", type="Gene"), Node(id="n2", type="Gene"))
    edges = (Edge(source="n1", target="n2", type="INTERACTS"),)
    graph = Graph(nodes=nodes, edges=edges)
    
    # Should not raise any exceptions
    validate_graph(graph, base_config)


def test_validate_duplicate_nodes_raises(base_config: GraphConfig) -> None:
    """Test that duplicate node IDs raise a ValueError."""
    nodes = (Node(id="n1", type="Gene"), Node(id="n1", type="Protein"))
    graph = Graph(nodes=nodes, edges=())
    
    with pytest.raises(ValueError, match="Duplicate node ID"):
        validate_graph(graph, base_config)


def test_validate_duplicate_edges_raises_directed(base_config: GraphConfig) -> None:
    """Test that duplicate edges raise ValueError in directed graphs."""
    nodes = (Node(id="n1", type="Gene"), Node(id="n2", type="Gene"))
    edges = (
        Edge(source="n1", target="n2", type="INTERACTS"),
        Edge(source="n1", target="n2", type="INTERACTS")
    )
    graph = Graph(nodes=nodes, edges=edges)
    
    with pytest.raises(ValueError, match="Parallel edges not allowed"):
        validate_graph(graph, base_config)


def test_validate_duplicate_edges_raises_undirected() -> None:
    """Test that reversed duplicate edges raise ValueError in undirected graphs."""
    config = GraphConfig(directed=False, weighted=True, allow_self_loops=False, allow_parallel_edges=False)
    nodes = (Node(id="n1", type="Gene"), Node(id="n2", type="Gene"))
    edges = (
        Edge(source="n1", target="n2", type="INTERACTS"),
        Edge(source="n2", target="n1", type="INTERACTS")
    )
    graph = Graph(nodes=nodes, edges=edges)
    
    with pytest.raises(ValueError, match="Parallel edges not allowed"):
        validate_graph(graph, config)


def test_validate_dangling_edges_raises(base_config: GraphConfig) -> None:
    """Test that edges referring to non-existent nodes raise ValueError."""
    nodes = (Node(id="n1", type="Gene"),)
    edges = (Edge(source="n1", target="n2", type="INTERACTS"),)
    graph = Graph(nodes=nodes, edges=edges)
    
    with pytest.raises(ValueError, match="Edge refers to non-existent node"):
        validate_graph(graph, base_config)


def test_validate_self_loop_raises(base_config: GraphConfig) -> None:
    """Test that self-loops raise ValueError when not allowed."""
    nodes = (Node(id="n1", type="Gene"),)
    edges = (Edge(source="n1", target="n1", type="INTERACTS"),)
    graph = Graph(nodes=nodes, edges=edges)
    
    with pytest.raises(ValueError, match="Self-loops not allowed"):
        validate_graph(graph, base_config)


def test_validate_self_loop_passes() -> None:
    """Test that self-loops pass validation when allowed."""
    config = GraphConfig(directed=True, weighted=True, allow_self_loops=True, allow_parallel_edges=False)
    nodes = (Node(id="n1", type="Gene"),)
    edges = (Edge(source="n1", target="n1", type="INTERACTS"),)
    graph = Graph(nodes=nodes, edges=edges)
    
    # Should not raise any exceptions
    validate_graph(graph, config)


def test_validate_invalid_weight() -> None:
    """Test that missing weights raise ValueError when weighted=True."""
    config = GraphConfig(directed=True, weighted=True, allow_self_loops=False, allow_parallel_edges=False)
    nodes = (Node(id="n1", type="Gene"), Node(id="n2", type="Gene"))
    edges = (Edge(source="n1", target="n2", type="INTERACTS", weight=-1.0),)
    graph = Graph(nodes=nodes, edges=edges)
    
    with pytest.raises(ValueError, match="Invalid weight"):
        validate_graph(graph, config)


def test_validate_empty_graph_warning(base_config: GraphConfig) -> None:
    """Test that validating an empty graph issues a warning."""
    graph = Graph(nodes=(), edges=())
    
    with pytest.warns(UserWarning, match="Graph is empty"):
        validate_graph(graph, base_config)
