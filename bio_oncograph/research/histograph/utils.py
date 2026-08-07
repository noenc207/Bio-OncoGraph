"""Utility functions for histopathology graph construction."""
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix between positions.

    Args:
        positions: A numpy array of shape (N, D) containing N positions in D dimensions.

    Returns:
        A numpy array of shape (N, N) containing the pairwise Euclidean distances.
    """
    logger.debug(f"Computing pairwise distances for {positions.shape[0]} points.")
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist_matrix


def knn_from_distances(dist_matrix: np.ndarray, k: int) -> np.ndarray:
    """Construct k-nearest neighbors graph from a distance matrix.

    Args:
        dist_matrix: A numpy array of shape (N, N) containing pairwise distances.
        k: The number of nearest neighbors to connect.

    Returns:
        A numpy array of shape (2, N*k) containing the edge indices (source, target).
    """
    num_nodes = dist_matrix.shape[0]
    logger.debug(f"Computing {k}-NN graph for {num_nodes} nodes.")
    
    sorted_indices = np.argsort(dist_matrix, axis=1)
    
    target_indices = sorted_indices[:, 1:k+1]
    source_indices = np.repeat(np.arange(num_nodes), min(k, max(0, num_nodes - 1)))
    
    if source_indices.size == 0 or target_indices.size == 0:
        return np.empty((2, 0), dtype=np.int64)

    edge_index = np.vstack((source_indices, target_indices.flatten()))
    return edge_index


def radius_from_distances(dist_matrix: np.ndarray, radius: float) -> np.ndarray:
    """Construct a radius graph from a distance matrix.

    Args:
        dist_matrix: A numpy array of shape (N, N) containing pairwise distances.
        radius: The distance threshold for creating an edge.

    Returns:
        A numpy array of shape (2, E) containing the edge indices (source, target).
    """
    logger.debug(f"Computing radius graph with radius {radius}.")
    
    mask = (dist_matrix <= radius) & (dist_matrix > 0.0)
    source_indices, target_indices = np.where(mask)
    
    edge_index = np.vstack((source_indices, target_indices))
    return edge_index


def count_connected_components(edge_index: np.ndarray, num_nodes: int) -> int:
    """Count the number of connected components in a graph using BFS.

    Args:
        edge_index: A numpy array of shape (2, E) containing edge indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        The number of connected components.
    """
    logger.debug(f"Counting connected components for graph with {num_nodes} nodes.")
    adj_list = {i: [] for i in range(num_nodes)}
    if edge_index.size > 0:
        for u, v in zip(edge_index[0], edge_index[1]):
            adj_list[u].append(v)
            adj_list[v].append(u)

    visited = np.zeros(num_nodes, dtype=bool)
    num_components = 0

    for i in range(num_nodes):
        if not visited[i]:
            num_components += 1
            queue = deque([i])
            visited[i] = True
            while queue:
                curr = queue.popleft()
                for neighbor in adj_list[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        
    return num_components


def compute_graph_density(num_nodes: int, num_edges: int, directed: bool) -> float:
    """Compute the density of a graph.

    Args:
        num_nodes: The number of nodes in the graph.
        num_edges: The number of edges in the graph.
        directed: A boolean indicating whether the graph is directed.

    Returns:
        The density of the graph, between 0.0 and 1.0.
    """
    logger.debug(f"Computing graph density: {num_nodes} nodes, {num_edges} edges, directed={directed}.")
    if num_nodes <= 1:
        return 0.0
    
    max_edges = num_nodes * (num_nodes - 1)
    if not directed:
        max_edges //= 2
        
    return float(num_edges) / max_edges if max_edges > 0 else 0.0


def remove_self_loops(edge_index: np.ndarray) -> np.ndarray:
    """Remove self-loops from the graph.

    Args:
        edge_index: A numpy array of shape (2, E) containing edge indices.

    Returns:
        A numpy array of shape (2, E') with self-loops removed.
    """
    logger.debug("Removing self loops.")
    if edge_index.size == 0:
        return edge_index
        
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Add self-loops to the graph.

    Args:
        edge_index: A numpy array of shape (2, E) containing edge indices.
        num_nodes: The total number of nodes in the graph.

    Returns:
        A numpy array of shape (2, E + num_nodes) containing edge indices with self-loops.
    """
    logger.debug(f"Adding self loops for {num_nodes} nodes.")
    loop_indices = np.arange(num_nodes)
    loops = np.vstack((loop_indices, loop_indices))
    
    if edge_index.size == 0:
        return loops
        
    edge_index_no_loops = remove_self_loops(edge_index)
    return np.hstack((edge_index_no_loops, loops))
