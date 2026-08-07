"""Edge strategies for histopathology graph construction."""
import abc
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

class EdgeStrategy(abc.ABC):
    """Abstract base class for edge construction strategies."""

    @abc.abstractmethod
    def compute_edges(
        self,
        positions: np.ndarray,
        config: 'HistographConfig'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Computes edges given node positions and configuration.

        Args:
            positions: Numpy array of node positions, shape (N, D).
            config: Configuration object containing edge strategy parameters.

        Returns:
            A tuple containing:
                - edge_index: Numpy array of shape (2, E).
                - edge_weights: Optional numpy array of shape (E,).
        """
        pass

class KNNEdgeStrategy(EdgeStrategy):
    """K-Nearest Neighbors edge construction strategy."""

    def compute_edges(
        self,
        positions: np.ndarray,
        config: 'HistographConfig'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Computes K-Nearest Neighbors edges.

        Args:
            positions: Numpy array of node positions, shape (N, D).
            config: Configuration object.

        Returns:
            Tuple of edge_index and edge_weights.
        """
        n_nodes = positions.shape[0]
        if n_nodes == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dist = np.sqrt(dist_sq)

        # Ensure we don't connect to self unless self_loops is enabled
        if not config.self_loops:
            np.fill_diagonal(dist, np.inf)

        # Determine number of neighbors to pick
        k = min(config.k_neighbors, n_nodes - 1 if not config.self_loops else n_nodes)
        if k <= 0:
             return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

        # Find k nearest neighbors for each node
        knn_indices = np.argsort(dist, axis=1)[:, :k]

        src_indices = np.repeat(np.arange(n_nodes), k)
        dst_indices = knn_indices.flatten()
        
        edge_index = np.stack([src_indices, dst_indices], axis=0)
        
        # Calculate weights (e.g., inverse distance)
        weights = 1.0 / (dist[src_indices, dst_indices] + 1e-8)

        if not config.directed:
            # Symmetrize
            edge_index_rev = np.stack([dst_indices, src_indices], axis=0)
            edge_index = np.concatenate([edge_index, edge_index_rev], axis=1)
            weights = np.concatenate([weights, weights], axis=0)
            # Remove duplicates
            # np.unique with axis=1 requires sorting or returns sorted array, 
            # we can use unique on contiguous array view or simple set logic, 
            # or just rely on np.unique output
            _, unique_idx = np.unique(edge_index, axis=1, return_index=True)
            edge_index = edge_index[:, unique_idx]
            weights = weights[unique_idx]
            
        return edge_index, weights

class RadiusEdgeStrategy(EdgeStrategy):
    """Radius-based edge construction strategy."""

    def compute_edges(
        self,
        positions: np.ndarray,
        config: 'HistographConfig'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Computes edges connecting nodes within a given radius.

        Args:
            positions: Numpy array of node positions, shape (N, D).
            config: Configuration object.

        Returns:
            Tuple of edge_index and edge_weights.
        """
        n_nodes = positions.shape[0]
        if n_nodes == 0:
             return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dist = np.sqrt(dist_sq)

        if not config.self_loops:
            np.fill_diagonal(dist, np.inf)

        src, dst = np.where(dist <= config.radius)
        edge_index = np.stack([src, dst], axis=0)
        weights = 1.0 / (dist[src, dst] + 1e-8)

        if not config.directed:
            # Undirected is naturally satisfied by distance symmetry
            pass

        return edge_index, weights

class FullyConnectedEdgeStrategy(EdgeStrategy):
    """Fully connected edge construction strategy."""

    def compute_edges(
        self,
        positions: np.ndarray,
        config: 'HistographConfig'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Computes a fully connected graph.

        Args:
            positions: Numpy array of node positions, shape (N, D).
            config: Configuration object.

        Returns:
            Tuple of edge_index and edge_weights.
        """
        n_nodes = positions.shape[0]
        if n_nodes == 0:
             return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
        
        src, dst = np.where(np.ones((n_nodes, n_nodes), dtype=bool))
        
        if not config.self_loops:
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
            
        edge_index = np.stack([src, dst], axis=0)
        
        diff = positions[src] - positions[dst]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        weights = 1.0 / (dist + 1e-8)

        return edge_index, weights
