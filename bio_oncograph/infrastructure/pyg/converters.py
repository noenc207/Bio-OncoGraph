"""Functions to convert PyTorch tensors into PyG representations."""

from typing import Tuple
import torch
import torch_geometric

def dense_to_sparse_edge_index(dense_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a dense adjacency matrix to a sparse edge_index and edge_attr.
    
    Args:
        dense_adj: Dense adjacency matrix of shape (N, N).
        
    Returns:
        Tuple containing edge_index (2, E) and edge_attr (E,).
    """
    if dense_adj.dim() != 2 or dense_adj.size(0) != dense_adj.size(1):
        raise ValueError("Dense adjacency matrix must be 2-dimensional and square.")
        
    indices = torch.nonzero(dense_adj, as_tuple=False)
    edge_index = indices.t().contiguous()
    edge_attr = dense_adj[indices[:, 0], indices[:, 1]]
    
    return edge_index, edge_attr

def sparse_to_dense_adj(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Converts a sparse edge_index to a dense adjacency matrix.
    
    Args:
        edge_index: Edge indices of shape (2, E).
        edge_attr: Edge attributes of shape (E,).
        num_nodes: Number of nodes in the graph.
        
    Returns:
        Dense adjacency matrix of shape (num_nodes, num_nodes).
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_attr.dtype, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = edge_attr
    return adj
