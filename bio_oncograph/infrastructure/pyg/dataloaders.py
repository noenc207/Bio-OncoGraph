"""Data loader wrappers for PyTorch Geometric."""

from typing import Any
import torch_geometric
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.data import Dataset, Data

def create_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, **kwargs: Any) -> DataLoader:
    """Creates a standard PyG DataLoader.
    
    Args:
        dataset: The PyG dataset to load from.
        batch_size: Number of graphs per batch.
        shuffle: Whether to shuffle the dataset.
        **kwargs: Additional arguments for torch_geometric.loader.DataLoader.
        
    Returns:
        A PyG DataLoader instance.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

def create_neighbor_loader(data: Data, num_neighbors: list, batch_size: int = 32, **kwargs: Any) -> NeighborLoader:
    """Creates a NeighborLoader for large graphs.
    
    Args:
        data: A PyG Data object representing a single large graph.
        num_neighbors: List specifying the number of neighbors to sample for each hop.
        batch_size: Number of nodes per batch.
        **kwargs: Additional arguments for torch_geometric.loader.NeighborLoader.
        
    Returns:
        A PyG NeighborLoader instance.
    """
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        **kwargs
    )
