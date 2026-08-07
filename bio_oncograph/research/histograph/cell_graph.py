"""Cell graph structures and builder."""
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np

from bio_oncograph.research.histograph.config import HistographConfig
from bio_oncograph.research.histograph.edges import KNNEdgeStrategy, RadiusEdgeStrategy, FullyConnectedEdgeStrategy

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CellNode:
    """Represents a cell node in the graph.

    Attributes:
        node_id: Unique identifier for the cell.
        centroid: (x, y) coordinates of the cell centroid.
        features: Feature vector of the cell.
        cell_type: Optional string indicating the cell type.
    """
    node_id: int
    centroid: Tuple[float, float]
    features: np.ndarray
    cell_type: Optional[str] = None

@dataclass(frozen=True)
class CellGraph:
    """Represents a graph of cell nodes.

    Attributes:
        nodes: Tuple of CellNode instances.
        edge_index: Numpy array of shape (2, E) representing edges.
        edge_weights: Optional numpy array of shape (E,) representing edge weights.
        metadata: Dictionary containing graph metadata.
    """
    nodes: Tuple[CellNode, ...]
    edge_index: np.ndarray
    edge_weights: Optional[np.ndarray]
    metadata: Dict[str, Any]

class CellGraphBuilder:
    """Builder for CellGraph."""

    def build(
        self,
        centroids: np.ndarray,
        features: np.ndarray,
        config: HistographConfig,
        cell_types: Optional[np.ndarray] = None
    ) -> CellGraph:
        """Builds a CellGraph from given data.

        Args:
            centroids: Array of shape (N, 2) containing cell coordinates.
            features: Array of shape (N, D) containing cell features.
            config: Configuration object.
            cell_types: Optional array of shape (N,) containing cell types.

        Returns:
            Constructed CellGraph.
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
            c_type = str(cell_types[i]) if cell_types is not None else None
            node = CellNode(
                node_id=i,
                centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                features=features[i],
                cell_type=c_type
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
            'graph_type': 'cell',
            'node_count': n_nodes,
            'edge_count': edge_index.shape[1] if edge_index is not None else 0
        }

        return CellGraph(
            nodes=nodes,
            edge_index=edge_index,
            edge_weights=edge_weights,
            metadata=metadata
        )
