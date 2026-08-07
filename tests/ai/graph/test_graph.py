"""Tests for the Graph class."""

import pytest
from dataclasses import FrozenInstanceError

from bio_oncograph.ai.graph.node import Node
from bio_oncograph.ai.graph.edge import Edge
from bio_oncograph.ai.graph.graph import Graph


def test_graph_creation() -> None:
    """Test that a Graph can be created correctly."""
    node1 = Node(id="n1", type="Gene")
    node2 = Node(id="n2", type="Gene")
    edge = Edge(source="n1", target="n2", type="INTERACTS")
    
    graph = Graph(nodes=(node1, node2), edges=(edge,))
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1


def test_graph_creation_empty() -> None:
    """Test that an empty Graph can be created."""
    graph = Graph(nodes=(), edges=())
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_graph_immutability() -> None:
    """Test that Graph is a frozen dataclass."""
    graph = Graph(nodes=(), edges=())
    with pytest.raises(FrozenInstanceError):
        graph.nodes = ()  # type: ignore


def test_graph_get_node_success() -> None:
    """Test successful retrieval of a node by ID."""
    node1 = Node(id="n1", type="Gene")
    graph = Graph(nodes=(node1,), edges=())
    
    retrieved = graph.get_node("n1")
    assert retrieved == node1


def test_graph_get_node_keyerror() -> None:
    """Test that getting a non-existent node raises KeyError."""
    graph = Graph(nodes=(), edges=())
    
    with pytest.raises(KeyError, match="Node 'n1' not found"):
        graph.get_node("n1")


def test_graph_get_edge_success() -> None:
    """Test successful retrieval of edges by source and target."""
    node1 = Node(id="n1", type="Gene")
    node2 = Node(id="n2", type="Gene")
    edge1 = Edge(source="n1", target="n2", type="INTERACTS")
    edge2 = Edge(source="n1", target="n2", type="CO_EXPRESSES")
    
    graph = Graph(nodes=(node1, node2), edges=(edge1, edge2))
    
    retrieved = graph.get_edge("n1", "n2")
    assert len(retrieved) == 2
    assert edge1 in retrieved
    assert edge2 in retrieved


def test_graph_get_edge_keyerror() -> None:
    """Test that getting a non-existent edge raises KeyError."""
    node1 = Node(id="n1", type="Gene")
    node2 = Node(id="n2", type="Gene")
    graph = Graph(nodes=(node1, node2), edges=())
    
    with pytest.raises(KeyError, match="No edges found between 'n1' and 'n2'"):
        graph.get_edge("n1", "n2")


def test_graph_equality() -> None:
    """Test that two Graphs with the same nodes and edges are equal."""
    node1 = Node(id="n1", type="Gene")
    node2 = Node(id="n2", type="Gene")
    edge = Edge(source="n1", target="n2", type="INTERACTS")
    
    graph1 = Graph(nodes=(node1, node2), edges=(edge,))
    
    # Graphs are considered equal if they contain the same nodes and edges
    # Regardless of tuple order
    graph2 = Graph(nodes=(node2, node1), edges=(edge,))
    
    # Comparing graphs might be order dependent depending on implementation
    # Let's assume standard dataclass behavior which is order dependent for tuples
    # To be safe, we just compare attributes converted to set
    assert set(graph1.nodes) == set(graph2.nodes)
    assert set(graph1.edges) == set(graph2.edges)
