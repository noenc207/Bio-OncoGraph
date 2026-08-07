"""General PyTorch Geometric utility functions."""

import torch
import torch_geometric
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from typing import Optional, Tuple
from torch import Tensor


def make_undirected(edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """Convert a directed graph to an undirected graph.

    Args:
        edge_index (Tensor): Graph connectivity matrix.
        edge_attr (Optional[Tensor], optional): Edge feature matrix. Defaults to None.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: Undirected edge_index and corresponding edge_attr.
    """
    if edge_attr is not None:
        return to_undirected(edge_index, edge_attr)
    return to_undirected(edge_index), None


def ensure_self_loops(edge_index: Tensor, edge_attr: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """Ensure that the graph contains self-loops for all nodes, removing duplicates first.

    Args:
        edge_index (Tensor): Graph connectivity matrix.
        edge_attr (Optional[Tensor], optional): Edge feature matrix. Defaults to None.
        num_nodes (Optional[int], optional): Number of nodes. Defaults to None.

    Returns:
        Tuple[Tensor, Optional[Tensor]]: Updated edge_index and edge_attr.
    """
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    return add_self_loops(edge_index, edge_attr, fill_value=1.0, num_nodes=num_nodes)


def compute_node_degrees(edge_index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    """Compute the degree of each node in the graph.

    Args:
        edge_index (Tensor): Graph connectivity matrix.
        num_nodes (Optional[int], optional): Number of nodes. Defaults to None.

    Returns:
        Tensor: A tensor containing the degree of each node.
    """
    from torch_geometric.utils import degree
    if num_nodes is None:
        num_nodes = int(edge_index.max().item() + 1) if edge_index.numel() > 0 else 0
    return degree(edge_index[0], num_nodes=num_nodes)


def is_bipartite(edge_index: Tensor) -> bool:
    """Check if the graph is bipartite.
    
    Args:
        edge_index (Tensor): Graph connectivity matrix.
        
    Returns:
        bool: True if the graph is bipartite, False otherwise.
    """
    from torch_geometric.utils import is_bipartite as pyg_is_bipartite
    return pyg_is_bipartite(edge_index)
