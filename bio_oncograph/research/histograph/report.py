"""Module for generating reports on graph structures."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphReport:
    """A report summarizing the structural properties of a graph.

    Attributes:
        graph_type: The type or level of the graph (e.g., cell, tissue).
        node_count: The total number of nodes.
        edge_count: The total number of edges.
        density: The density of the graph.
        avg_degree: The average degree of nodes.
        connected_components: The number of connected components.
        has_self_loops: Whether the graph contains self-loops.
        is_directed: Whether the graph is directed.
        metadata: Additional metadata associated with the graph.
    """

    graph_type: str
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    connected_components: int
    has_self_loops: bool
    is_directed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportGenerator:
    """Generator for creating graph reports."""

    def __init__(self) -> None:
        """Initialize the ReportGenerator."""
        logger.info("Initializing ReportGenerator.")

    def generate(self, graph: Any) -> GraphReport:
        """Generate a report for the given graph.

        Args:
            graph: The graph object to analyze. Expected to have attributes
                   or dictionary keys for structure.

        Returns:
            A GraphReport summarizing the graph.
        """
        logger.info("Generating report for graph.")
        
        if isinstance(graph, dict):
            graph_type = graph.get("type", "unknown")
            num_nodes = graph.get("num_nodes", 0)
            edge_index = graph.get("edge_index", np.empty((2, 0)))
            metadata = graph.get("metadata", {})
            is_directed = graph.get("is_directed", False)
        else:
            graph_type = getattr(graph, "type", type(graph).__name__)
            num_nodes = getattr(graph, "num_nodes", 0)
            edge_index = getattr(graph, "edge_index", np.empty((2, 0)))
            metadata = getattr(graph, "metadata", {})
            is_directed = getattr(graph, "is_directed", False)

        num_edges = edge_index.shape[1] if edge_index.ndim == 2 else 0

        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0

        max_edges = num_nodes * (num_nodes - 1)
        if not is_directed:
            max_edges //= 2
        density = float(num_edges) / max_edges if max_edges > 0 else 0.0

        has_self_loops = False
        if edge_index.size > 0:
            has_self_loops = bool(np.any(edge_index[0] == edge_index[1]))

        connected_components = 1 if num_nodes > 0 else 0

        report = GraphReport(
            graph_type=graph_type,
            node_count=num_nodes,
            edge_count=num_edges,
            density=density,
            avg_degree=avg_degree,
            connected_components=connected_components,
            has_self_loops=has_self_loops,
            is_directed=is_directed,
            metadata=metadata,
        )
        logger.info("Report generation complete.")
        return report
