"""
Graph data structure for Bio-OncoGraph.

Provides ``Graph`` — an immutable representation of a constructed
graph, containing nodes, edges, and graph-level metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from bio_oncograph.ai.graph.edge import Edge
from bio_oncograph.ai.graph.node import Node


@dataclass(frozen=True)
class Graph:
    """
    Immutable graph representation.

    Attributes:
        nodes: A tuple of ``Node`` objects.
        edges: A tuple of ``Edge`` objects.
        metadata: Dictionary of graph-level metadata (e.g.,
            construction parameters, source sample ID).
    """

    nodes: Tuple[Node, ...]
    edges: Tuple[Edge, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: Any) -> Node:
        """Retrieve a node by its ID. Raises KeyError if not found."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"Node '{node_id}' not found.")

    def get_edge(self, edge_id: Any) -> Edge:
        """Retrieve an edge by its ID. Raises KeyError if not found."""
        for e in self.edges:
            if e.id == edge_id:
                return e
        raise KeyError(f"Edge '{edge_id}' not found.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented
        return (
            self.nodes == other.nodes
            and self.edges == other.edges
            and self.metadata == other.metadata
        )
