"""Tests for the Node class."""

import pytest
from dataclasses import FrozenInstanceError

from bio_oncograph.ai.graph.node import Node


def test_node_creation() -> None:
    """Test that a Node can be created correctly."""
    node = Node(id="node_1", type="Gene", attributes={"symbol": "TP53"})
    assert node.id == "node_1"
    assert node.type == "Gene"
    assert node.attributes == {"symbol": "TP53"}


def test_node_creation_default_attributes() -> None:
    """Test Node creation with default attributes."""
    node = Node(id="node_1", type="Gene")
    assert node.id == "node_1"
    assert node.type == "Gene"
    assert node.attributes == {}


def test_node_immutability() -> None:
    """Test that Node is a frozen dataclass."""
    node = Node(id="node_1", type="Gene")
    with pytest.raises(FrozenInstanceError):
        node.id = "node_2"  # type: ignore


def test_node_equality() -> None:
    """Test that two Nodes with the same data are equal."""
    node1 = Node(id="node_1", type="Gene", attributes={"symbol": "TP53"})
    node2 = Node(id="node_1", type="Gene", attributes={"symbol": "TP53"})
    node3 = Node(id="node_2", type="Gene", attributes={"symbol": "TP53"})
    
    assert node1 == node2
    assert node1 != node3


def test_node_hash() -> None:
    """Test that Nodes can be hashed and used in sets/dicts."""
    node1 = Node(id="node_1", type="Gene", attributes={"symbol": "TP53"})
    node2 = Node(id="node_1", type="Gene", attributes={"symbol": "TP53"})
    
    assert hash(node1) == hash(node2)
    node_set = {node1, node2}
    assert len(node_set) == 1
