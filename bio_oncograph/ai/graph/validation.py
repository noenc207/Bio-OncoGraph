"""
Graph validation logic for Bio-OncoGraph.

Provides functions to validate graphs against configuration rules.
"""

from typing import List, Tuple

from bio_oncograph.ai.graph.config import GraphConfig
from bio_oncograph.ai.graph.graph import Graph


def validate_graph(
    graph: Graph, config: GraphConfig
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Validate a graph according to the given configuration.

    Args:
        graph: The ``Graph`` to validate.
        config: The ``GraphConfig`` with validation rules.

    Returns:
        A tuple ``(is_valid, warnings)``.
        Raises ValueError if a strict validation rule is violated.
    """
    warnings: List[str] = []

    _validate_unique_node_ids(graph)
    _validate_unique_edge_ids(graph)
    _validate_dangling_edges(graph)
    
    if not config.allow_self_loop:
        _validate_no_self_loops(graph)
        
    if config.weighted:
        _validate_edge_weights(graph)
        
    _validate_no_duplicate_edges(graph, config.directed)

    if not graph.nodes:
        warnings.append("Graph is empty (0 nodes).")
        
    return True, tuple(warnings)


def _validate_unique_node_ids(graph: Graph) -> None:
    seen = set()
    for n in graph.nodes:
        if n.id in seen:
            raise ValueError(f"Duplicate node ID found: {n.id}")
        seen.add(n.id)


def _validate_unique_edge_ids(graph: Graph) -> None:
    seen = set()
    for e in graph.edges:
        if e.id in seen:
            raise ValueError(f"Duplicate edge ID found: {e.id}")
        seen.add(e.id)


def _validate_dangling_edges(graph: Graph) -> None:
    node_ids = {n.id for n in graph.nodes}
    for e in graph.edges:
        if e.source_id not in node_ids:
            raise ValueError(
                f"Edge {e.id} references missing source node {e.source_id}"
            )
        if e.target_id not in node_ids:
            raise ValueError(
                f"Edge {e.id} references missing target node {e.target_id}"
            )


def _validate_no_self_loops(graph: Graph) -> None:
    for e in graph.edges:
        if e.source_id == e.target_id:
            raise ValueError(f"Self-loop detected on node {e.source_id} (Edge {e.id})")


def _validate_edge_weights(graph: Graph) -> None:
    for e in graph.edges:
        if not isinstance(e.weight, (int, float)):
            raise ValueError(f"Edge {e.id} has non-numeric weight {e.weight}")


def _validate_no_duplicate_edges(graph: Graph, directed: bool) -> None:
    seen = set()
    for e in graph.edges:
        u, v = e.source_id, e.target_id
        if not directed and str(u) > str(v):  # normalize undirected representation
            u, v = v, u
            
        pair = (u, v)
        if pair in seen:
            raise ValueError(f"Duplicate edge detected between {u} and {v}")
        seen.add(pair)
