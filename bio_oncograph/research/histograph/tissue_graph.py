"""Tissue graph structures and builder."""
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np

from bio_oncograph.research.histograph.config import HistographConfig
from bio_oncograph.research.histograph.edges import KNNEdgeStrategy, RadiusEdgeStrategy, FullyConnectedEdgeStrategy

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TissueNode:
    """Represents a tissue patch node in the graph.

    Attributes:
        node_id: Unique identifier for the tissue node.
        centroid: (x, y) coordinates of the tissue centroid.
        features: Feature vector of the tissue.
        region_label: Optional string indicating the region label.
    """
    node_id: int
    centroid: Tuple[float, float]
    features: np.ndarray
    region_label: Optional[str] = None

@dataclass(frozen=True)
class TissueGraph:
    """Represents a graph of tissue nodes.

    Attributes:
        nodes: Tuple of TissueNode instances.
        edge_index: Numpy array of shape (2, E) representing edges.
        edge_weights: Optional numpy array of shape (E,) representing edge weights.
        metadata: Dictionary containing graph metadata.
    """
    nodes: Tuple[TissueNode, ...]
    edge_index: np.ndarray
    edge_weights: Optional[np.ndarray]
    metadata: Dict[str, Any]

class TissueGraphBuilder:
    """Builder for TissueGraph."""

    def build(
        self,
        centroids: np.ndarray,
        features: np.ndarray,
        config: HistographConfig,
        region_labels: Optional[np.ndarray] = None
    ) -> TissueGraph:
        """Builds a TissueGraph from given data.

        Args:
            centroids: Array of shape (N, 2) containing tissue coordinates.
            features: Array of shape (N, D) containing tissue features.
            config: Configuration object.
            region_labels: Optional array of shape (N,) containing region labels.

        Returns:
            Constructed TissueGraph.
        """
        n_nodes = centroids.shape[0]
        if n_nodes < config.min_nodes:
            logger.warning(
                "Number of nodes (%d) is less than min_nodes (%d).",
                n_nodes, config.min_nodes
            )
        
        # Build nodes
        nodes_list = []
        for i in range(n_nodes):
            r_label = str(region_labels[i]) if region_labels is not None else None
            node = TissueNode(
                node_id=i,
                centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                features=features[i],
                region_label=r_label
            )
            nodes_list.append(node)
            
        nodes = tuple(nodes_list)

        # Build edges
        if config.edge_strategy == 'knn':
            strategy = KNNEdgeStrategy()
        elif config.edge_strategy == 'radius':
            strategy = RadiusEdgeStrategy()
        else:
            strategy = FullyConnectedEdgeStrategy()

        edge_index, edge_weights = strategy.compute_edges(centroids, config)
        
        metadata = {
            'graph_type': 'tissue',
            'node_count': n_nodes,
            'edge_count': edge_index.shape[1] if edge_index is not None else 0
        }

        return TissueGraph(
            nodes=nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            metadata=metadata
        )
