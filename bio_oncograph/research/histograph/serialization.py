"""Module for serializing and deserializing graphs."""
import json
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class GraphSerializer:
    """Handles serialization of simple graphs to and from JSON."""

    def __init__(self) -> None:
        """Initialize the GraphSerializer."""
        self.version = "1.0"
        logger.info("Initializing GraphSerializer.")

    def save(self, graph: Any, path: str) -> None:
        """Save a graph to a JSON file.

        Args:
            graph: The graph object to save.
            path: The file path to save to.
        """
        logger.info(f"Saving graph to {path}.")
        
        if isinstance(graph, dict):
            nodes = graph.get("nodes", [])
            edge_index = graph.get("edge_index", np.empty((2, 0)))
            metadata = graph.get("metadata", {})
        else:
            nodes = getattr(graph, "nodes", [])
            edge_index = getattr(graph, "edge_index", np.empty((2, 0)))
            metadata = getattr(graph, "metadata", {})

        if isinstance(edge_index, np.ndarray):
            edge_list = edge_index.tolist()
        else:
            edge_list = list(edge_index)

        data = {
            "version": self.version,
            "nodes": nodes,
            "edge_index": edge_list,
            "metadata": metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> Dict[str, Any]:
        """Load a graph dictionary from a JSON file.

        Args:
            path: The file path to load from.

        Returns:
            A dictionary containing graph data.
        """
        logger.info(f"Loading graph from {path}.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        loaded_version = data.get("version", "unknown")
        logger.debug(f"Loaded graph with format version: {loaded_version}")
        
        return data


class HierarchySerializer:
    """Handles serialization of hierarchical graphs."""

    def __init__(self) -> None:
        """Initialize the HierarchySerializer."""
        self.version = "1.0"
        logger.info("Initializing HierarchySerializer.")

    def save(self, hierarchical_graph: Any, path: str) -> None:
        """Save a hierarchical graph to a JSON file.

        Args:
            hierarchical_graph: The HierarchicalGraph object to save.
            path: The file path to save to.
        """
        logger.info(f"Saving hierarchical graph to {path}.")
        
        cross_level_edges = hierarchical_graph.cross_level_edges
        if isinstance(cross_level_edges, np.ndarray):
            cle_list = cross_level_edges.tolist()
        else:
            cle_list = list(cross_level_edges)

        data = {
            "version": self.version,
            "cross_level_edges": cle_list,
            "metadata": hierarchical_graph.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> Dict[str, Any]:
        """Load a hierarchical graph dict from a JSON file.

        Args:
            path: The file path to load from.

        Returns:
            A dictionary containing hierarchical graph data.
        """
        logger.info(f"Loading hierarchical graph from {path}.")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data
