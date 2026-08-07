"""Tests for the Edge class."""

import pytest
from dataclasses import FrozenInstanceError

from bio_oncograph.ai.graph.edge import Edge


def test_edge_creation() -> None:
    """Test that an Edge can be created correctly."""
    edge = Edge(source="node_1", target="node_2", type="INTERACTS_WITH", weight=0.8, attributes={"evidence": "experimental"})
    assert edge.source == "node_1"
    assert edge.target == "node_2"
    assert edge.type == "INTERACTS_WITH"
    assert edge.weight == 0.8
    assert edge.attributes == {"evidence": "experimental"}


def test_edge_default_weight() -> None:
    """Test Edge creation with default weight and attributes."""
    edge = Edge(source="node_1", target="node_2", type="INTERACTS_WITH")
    assert edge.source == "node_1"
    assert edge.target == "node_2"
    assert edge.type == "INTERACTS_WITH"
    assert edge.weight == 1.0
    assert edge.attributes == {}


def test_edge_immutability() -> None:
    """Test that Edge is a frozen dataclass."""
    edge = Edge(source="node_1", target="node_2", type="INTERACTS_WITH")
    with pytest.raises(FrozenInstanceError):
        edge.weight = 0.5  # type: ignore


def test_edge_equality() -> None:
    """Test that two Edges with the same data are equal."""
    edge1 = Edge(source="node_1", target="node_2", type="INTERACTS_WITH", weight=0.8)
    edge2 = Edge(source="node_1", target="node_2", type="INTERACTS_WITH", weight=0.8)
    edge3 = Edge(source="node_2", target="node_1", type="INTERACTS_WITH", weight=0.8)
    
    assert edge1 == edge2
    assert edge1 != edge3


def test_edge_hash() -> None:
    """Test that Edges can be hashed and used in sets/dicts."""
    edge1 = Edge(source="node_1", target="node_2", type="INTERACTS_WITH", weight=0.8)
    edge2 = Edge(source="node_1", target="node_2", type="INTERACTS_WITH", weight=0.8)
    
    assert hash(edge1) == hash(edge2)
    edge_set = {edge1, edge2}
    assert len(edge_set) == 1
