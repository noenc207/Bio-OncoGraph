"""
Graph statistics container for Bio-OncoGraph.

Provides ``GraphStatistics`` and functions to compute them.
"""

from dataclasses import dataclass

from bio_oncograph.ai.graph.config import GraphConfig
from bio_oncograph.ai.graph.graph import Graph


@dataclass(frozen=True)
class GraphStatistics:
    """
    Immutable container for calculated graph statistics.

    Attributes:
        num_nodes: Total number of nodes.
        num_edges: Total number of edges.
        average_degree: Average number of edges per node.
        density: Graph density (actual edges / possible edges).
        connected_components: Number of connected components
            (treating the graph as undirected for connectivity).
        isolated_nodes: Number of nodes with degree 0.
    """

    num_nodes: int
    num_edges: int
    average_degree: float
    density: float
    connected_components: int
    isolated_nodes: int


def compute_statistics(
    graph: Graph, config: GraphConfig
) -> GraphStatistics:
    """
    Compute basic structural statistics for a graph.

    Args:
        graph: The constructed ``Graph``.
        config: The ``GraphConfig`` (used to determine if directed).

    Returns:
        A ``GraphStatistics`` object.
    """
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)

    if num_nodes == 0:
        return GraphStatistics(0, 0, 0.0, 0.0, 0, 0)

    # Calculate degrees
    degrees = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        if e.source_id in degrees:
            degrees[e.source_id] += 1
        if e.target_id in degrees and not config.directed:
            if e.source_id != e.target_id:
                degrees[e.target_id] += 1

    total_degrees = sum(degrees.values())
    average_degree = total_degrees / num_nodes

    # Calculate density
    if num_nodes <= 1:
        density = 0.0
    else:
        possible_edges = num_nodes * (num_nodes - 1)
        if not config.directed:
            possible_edges //= 2
        
        # Self-loops alter standard density slightly, but for
        # generic graphs, standard formula is E / (N*(N-1)[/2])
        density = num_edges / possible_edges

    isolated_nodes = sum(1 for d in degrees.values() if d == 0)

    # Calculate connected components (undirected traversal)
    adj = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        if e.source_id in adj and e.target_id in adj:
            adj[e.source_id].add(e.target_id)
            adj[e.target_id].add(e.source_id)

    visited = set()
    components = 0
    for n in graph.nodes:
        if n.id not in visited:
            components += 1
            # BFS/DFS
            stack = [n.id]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    stack.extend(adj[curr] - visited)

    return GraphStatistics(
        num_nodes=num_nodes,
        num_edges=num_edges,
        average_degree=average_degree,
        density=density,
        connected_components=components,
        isolated_nodes=isolated_nodes,
    )
