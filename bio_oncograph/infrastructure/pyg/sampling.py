"""Abstractions for Graph Sampling using PyG utilities."""

import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader, ClusterLoader, ClusterData
from torch_geometric.data import Data
from typing import Any, List


def create_neighbor_sampler(
    data: Data,
    num_neighbors: List[int],
    batch_size: int = 128,
    shuffle: bool = True,
    **kwargs: Any
) -> NeighborLoader:
    """Create a NeighborLoader for neighbor sampling.

    Args:
        data (Data): The graph data object.
        num_neighbors (List[int]): The number of neighbors to sample for each hop.
        batch_size (int, optional): The batch size. Defaults to 128.
        shuffle (bool, optional): If set to True, the data will be shuffled. Defaults to True.
        **kwargs: Additional keyword arguments for NeighborLoader.

    Returns:
        NeighborLoader: The neighbor sampler loader.
    """
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def create_subgraph_sampler(
    data: Data,
    num_parts: int,
    batch_size: int = 1,
    shuffle: bool = True,
    **kwargs: Any
) -> ClusterLoader:
    """Create a ClusterLoader for subgraph sampling.

    Args:
        data (Data): The graph data object.
        num_parts (int): Number of partitions.
        batch_size (int, optional): The batch size. Defaults to 1.
        shuffle (bool, optional): If set to True, the data will be shuffled. Defaults to True.
        **kwargs: Additional keyword arguments for ClusterLoader.

    Returns:
        ClusterLoader: The subgraph sampler loader.
    """
    cluster_data = ClusterData(data, num_parts=num_parts, save_dir=None)
    return ClusterLoader(
        cluster_data,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def create_random_walk_sampler(
    data: Data,
    walk_length: int,
    context_size: int,
    walks_per_node: int = 1,
    p: float = 1.0,
    q: float = 1.0,
    num_negative_samples: int = 1,
    batch_size: int = 128,
    shuffle: bool = True,
    **kwargs: Any
) -> Any:
    """Create a Node2Vec model which includes random walk sampling.

    Args:
        data (Data): The graph data object.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for positive samples.
        walks_per_node (int, optional): The number of walks to sample for each node. Defaults to 1.
        p (float, optional): Likelihood of immediately revisiting a node in the walk. Defaults to 1.0.
        q (float, optional): Control parameter to interpolate between breadth-first strategy and depth-first strategy. Defaults to 1.0.
        num_negative_samples (int, optional): The number of negative samples to use for each positive sample. Defaults to 1.
        batch_size (int, optional): The batch size. Defaults to 128.
        shuffle (bool, optional): If set to True, the data will be shuffled. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        Node2Vec: The random walk sampler and model.
    """
    from torch_geometric.nn import Node2Vec
    return Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=kwargs.get("embedding_dim", 64),
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_negative_samples=num_negative_samples,
        sparse=kwargs.get("sparse", True)
    )
