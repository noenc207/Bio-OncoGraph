"""Validation utilities for PyG Data objects."""

import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import List, Optional


def validate_data_object(data: Data, check_features: bool = True) -> List[str]:
    """Validate a PyG Data object for common issues.

    Checks include edge_index shapes, missing features, disconnected nodes, and corrupted metadata.

    Args:
        data (Data): The PyG Data object to validate.
        check_features (bool, optional): Whether to check for node features (x). Defaults to True.

    Returns:
        List[str]: A list of error messages. Empty list if validation passes.
    """
    errors: List[str] = []

    if not isinstance(data, Data):
        errors.append("Object is not a valid PyG Data instance.")
        return errors

    if data.edge_index is not None:
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            errors.append(f"Invalid edge_index shape: {data.edge_index.shape}. Expected (2, num_edges).")
        elif data.num_nodes is not None and data.edge_index.numel() > 0:
            max_index = data.edge_index.max().item()
            if max_index >= data.num_nodes:
                errors.append(f"edge_index contains out-of-bounds node index: {max_index}. num_nodes: {data.num_nodes}.")

    if check_features:
        if data.x is None:
            errors.append("Node features (x) are missing.")
        elif data.num_nodes is not None and data.x.size(0) != data.num_nodes:
            errors.append(f"Mismatch between number of nodes ({data.num_nodes}) and feature size ({data.x.size(0)}).")

    if data.edge_index is not None and data.num_nodes is not None:
        try:
            from torch_geometric.utils import degree
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes) + degree(data.edge_index[1], num_nodes=data.num_nodes)
            if (deg == 0).any():
                errors.append("Graph contains isolated/disconnected nodes.")
        except ImportError:
            pass

    return errors


def validate_batch_consistency(batch: Batch) -> List[str]:
    """Validate the consistency of a PyG Batch object.

    Args:
        batch (Batch): The PyG Batch object to validate.

    Returns:
        List[str]: A list of error messages. Empty list if validation passes.
    """
    errors: List[str] = []
    
    if not isinstance(batch, Batch):
        errors.append("Object is not a valid PyG Batch instance.")
        return errors
        
    if batch.batch is None:
        errors.append("Batch assignment vector (batch) is missing.")
    elif batch.x is not None and batch.batch.size(0) != batch.x.size(0):
        errors.append(f"Mismatch between batch vector size ({batch.batch.size(0)}) and feature size ({batch.x.size(0)}).")
        
    if batch.ptr is None:
        errors.append("Batch pointer vector (ptr) is missing.")
        
    return errors
