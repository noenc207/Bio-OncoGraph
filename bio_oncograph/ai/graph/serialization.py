"""
Graph serialization logic for Bio-OncoGraph.

Supports serialization of ``Graph`` to dictionaries and deserialization
back to ``Graph``.
"""

from typing import Any, Dict

from bio_oncograph.ai.graph.edge import Edge
from bio_oncograph.ai.graph.graph import Graph
from bio_oncograph.ai.graph.node import Node


def serialize_graph(
    graph: Graph, version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Serialize a ``Graph`` into a JSON-compatible dictionary.

    Args:
        graph: The ``Graph`` to serialize.
        version: The schema version string.

    Returns:
        A dictionary representation of the graph.
    """
    return {
        "version": version,
        "metadata": graph.metadata.copy(),
        "nodes": [
            {
                "id": n.id,
                "features": n.features.copy(),
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "target_id": e.target_id,
                "weight": e.weight,
                "features": e.features.copy(),
            }
            for e in graph.edges
        ],
    }


def deserialize_graph(data: Dict[str, Any]) -> Graph:
    """
    Deserialize a dictionary into a ``Graph``.

    Args:
        data: The dictionary representation.

    Returns:
        A reconstructed ``Graph`` object.
    
    Raises:
        KeyError: If required keys are missing in the data.
    """
    nodes = tuple(
        Node(id=n["id"], features=n.get("features", {}))
        for n in data["nodes"]
    )
    
    edges = tuple(
        Edge(
            id=e["id"],
            source_id=e["source_id"],
            target_id=e["target_id"],
            weight=e.get("weight", 1.0),
            features=e.get("features", {}),
        )
        for e in data["edges"]
    )
    
    return Graph(
        nodes=nodes,
        edges=edges,
        metadata=data.get("metadata", {}).copy(),
    )
