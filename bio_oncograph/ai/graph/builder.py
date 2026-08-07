"""
Graph Construction Builder for Bio-OncoGraph.

Provides ``GraphBuilder`` — the top-level API for creating, validating,
and reporting on graphs from preprocessed data.
"""

import logging
from typing import Any, Dict, List, Optional

from bio_oncograph.ai.graph.config import GraphConfig
from bio_oncograph.ai.graph.edge import Edge
from bio_oncograph.ai.graph.graph import Graph
from bio_oncograph.ai.graph.node import Node
from bio_oncograph.ai.graph.report import GraphReport
from bio_oncograph.ai.graph.serialization import (
    deserialize_graph,
    serialize_graph,
)
from bio_oncograph.ai.graph.statistics import compute_statistics
from bio_oncograph.ai.graph.validation import validate_graph

logger = logging.getLogger("BioOncoGraph")


class GraphBuilder:
    """
    Engine to build and validate graphs.

    Args:
        config: Configuration for graph construction.
    """

    def __init__(self, config: GraphConfig = GraphConfig()):
        self._config = config

    @property
    def config(self) -> GraphConfig:
        """Return the active configuration."""
        return self._config

    def build(
        self,
        nodes: List[Node],
        edges: List[Edge],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """
        Build an immutable Graph object from lists of nodes and edges.
        Does not validate (use ``report`` or ``validate`` for that).

        Args:
            nodes: List of ``Node`` objects.
            edges: List of ``Edge`` objects.
            metadata: Optional graph metadata.

        Returns:
            An immutable ``Graph``.
        """
        if metadata is None:
            metadata = {}
        # Ensure connectivity rule is preserved
        meta = metadata.copy()
        if "connectivity_rule" not in meta:
            meta["connectivity_rule"] = self._config.connectivity_rule

        graph = Graph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            metadata=meta,
        )
        logger.info(
            "Built graph with %d nodes and %d edges.",
            len(nodes),
            len(edges),
        )
        return graph

    def validate(self, graph: Graph) -> bool:
        """
        Validate a graph.

        Args:
            graph: The ``Graph`` to validate.

        Returns:
            ``True`` if valid, ``False`` if a ValueError is raised during
            validation rules checking.
        """
        try:
            validate_graph(graph, self._config)
            return True
        except ValueError as e:
            logger.warning("Graph validation failed: %s", e)
            return False

    def statistics(self, graph: Graph) -> Any:
        """
        Compute statistics for a graph.

        Args:
            graph: The ``Graph``.

        Returns:
            A ``GraphStatistics`` object.
        """
        return compute_statistics(graph, self._config)

    def serialize(self, graph: Graph) -> Dict[str, Any]:
        """
        Serialize a graph to a dictionary.

        Args:
            graph: The ``Graph``.

        Returns:
            Dictionary representation.
        """
        return serialize_graph(graph, self._config.serialization_version)

    def deserialize(self, data: Dict[str, Any]) -> Graph:
        """
        Deserialize a dictionary into a graph.

        Args:
            data: Dictionary representation.

        Returns:
            A ``Graph``.
        """
        return deserialize_graph(data)

    def report(self, graph: Graph) -> GraphReport:
        """
        Validate and generate a full report for a graph.

        Args:
            graph: The ``Graph``.

        Returns:
            A ``GraphReport`` containing the graph, validity, stats,
            and any warnings.
        """
        try:
            is_valid, warnings = validate_graph(graph, self._config)
        except ValueError as e:
            is_valid = False
            warnings = (str(e),)

        stats = compute_statistics(graph, self._config)
        
        return GraphReport(
            graph=graph,
            is_valid=is_valid,
            statistics=stats,
            warnings=warnings,
            metadata={"config": self.serialize_config()},
        )

    def serialize_config(self) -> Dict[str, Any]:
        """Return the config as a dictionary for metadata."""
        return {
            "directed": self._config.directed,
            "allow_self_loop": self._config.allow_self_loop,
            "weighted": self._config.weighted,
            "connectivity_rule": self._config.connectivity_rule,
            "serialization_version": self._config.serialization_version,
        }
