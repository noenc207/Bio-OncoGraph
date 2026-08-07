"""Region graph structures and builder."""
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np

from bio_oncograph.research.histograph.config import HistographConfig
from bio_oncograph.research.histograph.edges import KNNEdgeStrategy, RadiusEdgeStrategy, FullyConnectedEdgeStrategy

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RegionNode:
    """Represents a region node in the graph.

    Attributes:
        node_id: Unique identifier for the region node.
        bounding_box: (min_x, min_y, max_x, max_y) coordinates of the bounding box.
        features: Feature vector of the region.
        region_type: Optional string indicating the region type.
    """
    node_id: int
    bounding_box: Tuple[int, int, int, int]
    features: np.ndarray
    region_type: Optional[str] = None

@dataclass(frozen=True)
class RegionGraph:
    """Represents a graph of region nodes.

    Attributes:
        nodes: Tuple of RegionNode instances.
        edge_index: Numpy array of shape (2, E) representing edges.
        edge_weights: Optional numpy array of shape (E,) representing edge weights.
        metadata: Dictionary containing graph metadata.
    """
    nodes: Tuple[RegionNode, ...]
    edge_index: np.ndarray
    edge_weights: Optional[np.ndarray]
    metadata: Dict[str, Any]

class RegionGraphBuilder:
    """Builder for RegionGraph."""

    def build(
        self,
        bboxes: np.ndarray,
        features: np.ndarray,
        config: HistographConfig,
        region_types: Optional[np.ndarray] = None
    ) -> RegionGraph:
        """Builds a RegionGraph from given data.

        Args:
            bboxes: Array of shape (N, 4) containing region bounding boxes.
            features: Array of shape (N, D) containing region features.
            config: Configuration object.
            region_types: Optional array of shape (N,) containing region types.

        Returns:
            Constructed RegionGraph.
        """
        n_nodes = bboxes.shape[0]
        if n_nodes < config.min_nodes:
            logger.warning(
                "Number of nodes (%d) is less than min_nodes (%d).",
                n_nodes, config.min_nodes
            )
        
        # Build nodes
        nodes_list = []
        centroids_list = []
        for i in range(n_nodes):
            r_type = str(region_types[i]) if region_types is not None else None
            min_x, min_y, max_x, max_y = bboxes[i]
            node = RegionNode(
                node_id=i,
                bounding_box=(int(min_x), int(min_y), int(max_x), int(max_y)),
                features=features[i],
                region_type=r_type
            )
            nodes_list.append(node)
            
            # Compute centroid for edge generation
            c_x = (min_x + max_x) / 2.0
            c_y = (min_y + max_y) / 2.0
            centroids_list.append([c_x, c_y])
            
        nodes = tuple(nodes_list)
        centroids = np.array(centroids_list)

        # Build edges
        if config.edge_strategy == 'knn':
            strategy = KNNEdgeStrategy()
        elif config.edge_strategy == 'radius':
            strategy = RadiusEdgeStrategy()
        else:
            strategy = FullyConnectedEdgeStrategy()

        # Compute edges using centroids of the bounding boxes
        edge_index, edge_weights = strategy.compute_edges(centroids, config)
        
        metadata = {
            'graph_type': 'region',
            'node_count': n_nodes,
            'edge_count': edge_index.shape[1] if edge_index is not None else 0
        }

        return RegionGraph(
            nodes=nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            metadata=metadata
        )
