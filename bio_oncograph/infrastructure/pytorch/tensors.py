"""Tensor conversion utilities for graph structures."""
import torch
from typing import Any, Dict, List, Optional


def create_tensor(data: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """Creates a PyTorch tensor from given data.

    Args:
        data (Any): The data to convert to a tensor (list, numpy array, etc.).
        dtype (Optional[torch.dtype]): The desired data type of the tensor.
        device (Optional[torch.device]): The device to place the tensor on.

    Returns:
        torch.Tensor: The created PyTorch tensor.
    """
    return torch.tensor(data, dtype=dtype, device=device)


def features_to_tensor(features: List[List[float]], device: Optional[torch.device] = None) -> torch.Tensor:
    """Converts a list of feature vectors into a 2D PyTorch tensor.

    Args:
        features (List[List[float]]): A list of feature vectors.
        device (Optional[torch.device]): The device to place the tensor on.

    Returns:
        torch.Tensor: A 2D float32 PyTorch tensor.
    """
    return torch.tensor(features, dtype=torch.float32, device=device)


def edges_to_tensor(edges: List[tuple[int, int]], device: Optional[torch.device] = None) -> torch.Tensor:
    """Converts a list of edges into a 2D edge index PyTorch tensor.

    Args:
        edges (List[tuple[int, int]]): A list of edges represented as (source, target) tuples.
        device (Optional[torch.device]): The device to place the tensor on.

    Returns:
        torch.Tensor: A 2D int64 PyTorch tensor of shape (2, num_edges).
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.int64, device=device)
    
    # PyG expects shape (2, num_edges)
    src, dst = zip(*edges)
    return torch.tensor([list(src), list(dst)], dtype=torch.int64, device=device)

def labels_to_tensor(labels: List[int], device: Optional[torch.device] = None) -> torch.Tensor:
    """Converts a list of integer labels into a 1D PyTorch tensor.

    Args:
        labels (List[int]): A list of class labels.
        device (Optional[torch.device]): The device to place the tensor on.

    Returns:
        torch.Tensor: A 1D int64 PyTorch tensor.
    """
    return torch.tensor(labels, dtype=torch.int64, device=device)
