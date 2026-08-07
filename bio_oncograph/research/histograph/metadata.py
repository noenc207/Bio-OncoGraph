"""Metadata structures for histopathology graph construction."""
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass(frozen=True)
class GraphMetadata:
    """Metadata for constructed graphs.

    Attributes:
        slide_id: Identifier of the whole slide image.
        patch_id: Identifier of the patch within the slide.
        graph_type: Type of the graph ('cell', 'tissue', 'region').
        node_count: Number of nodes in the graph.
        edge_count: Number of edges in the graph.
        feature_schema: Tuple describing the features (e.g., ('area', 'mean_intensity')).
        edge_schema: Description of how edges were formed (e.g., 'knn').
        construction_params: Dictionary of parameters used during construction.
        version: Version of the construction algorithm.
        timestamp: ISO format timestamp of when the graph was created.
    """
    slide_id: str
    patch_id: str
    graph_type: str
    node_count: int
    edge_count: int
    feature_schema: Tuple[str, ...]
    edge_schema: str
    construction_params: Dict[str, Any]
    version: str
    timestamp: str
