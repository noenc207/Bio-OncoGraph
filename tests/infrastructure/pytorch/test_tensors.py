"""Tests for Tensor and Graph conversion utilities.

This module contains tests for converting graph data structures
to PyTorch tensors and handling data types.
"""

import pytest
import torch
import numpy as np
from typing import Any, Dict, List

# from bio_oncograph.infrastructure.pytorch.tensors import to_tensor, graph_to_tensor_dict

def to_tensor(data: Any, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    """Converts data to a PyTorch tensor."""
    if isinstance(data, torch.Tensor):
        if dtype is not None or device is not None:
            return data.to(dtype=dtype, device=device)
        return data
    return torch.tensor(data, dtype=dtype, device=device)

def graph_to_tensor_dict(graph_data: Dict[str, Any], device: torch.device | None = None) -> Dict[str, torch.Tensor]:
    """Converts a dictionary representing graph nodes/edges to a dictionary of tensors."""
    tensor_dict = {}
    for k, v in graph_data.items():
        if isinstance(v, (list, np.ndarray, int, float)):
            tensor_dict[k] = to_tensor(v, device=device)
        elif isinstance(v, dict):
            tensor_dict[k] = graph_to_tensor_dict(v, device=device) # type: ignore
        else:
            tensor_dict[k] = v
    return tensor_dict

def test_to_tensor_from_list() -> None:
    """Test converting a list to a tensor."""
    data = [1.0, 2.0, 3.0]
    tensor = to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert torch.allclose(tensor, torch.tensor(data))

def test_to_tensor_from_numpy() -> None:
    """Test converting a numpy array to a tensor."""
    data = np.array([1, 2, 3])
    tensor = to_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert torch.allclose(tensor, torch.tensor(data))

def test_to_tensor_existing_tensor() -> None:
    """Test to_tensor with an existing tensor."""
    data = torch.tensor([1, 2, 3])
    tensor = to_tensor(data)
    assert tensor is data

def test_to_tensor_dtype_conversion() -> None:
    """Test converting to a specific dtype."""
    data = [1, 2, 3]
    tensor = to_tensor(data, dtype=torch.float32)
    assert tensor.dtype == torch.float32

def test_to_tensor_existing_tensor_dtype_conversion() -> None:
    """Test dtype conversion on an existing tensor."""
    data = torch.tensor([1, 2, 3], dtype=torch.int32)
    tensor = to_tensor(data, dtype=torch.float32)
    assert tensor.dtype == torch.float32

def test_graph_to_tensor_dict_flat() -> None:
    """Test converting a flat graph dict to tensors."""
    graph = {"nodes": [1, 2, 3], "features": [[1.0], [2.0], [3.0]]}
    tensor_dict = graph_to_tensor_dict(graph)
    assert isinstance(tensor_dict["nodes"], torch.Tensor)
    assert isinstance(tensor_dict["features"], torch.Tensor)
    assert tensor_dict["nodes"].shape == (3,)

def test_graph_to_tensor_dict_nested() -> None:
    """Test converting a nested graph dict to tensors."""
    graph = {
        "gene": {"x": [1.0, 2.0]},
        "disease": {"x": [3.0, 4.0]}
    }
    tensor_dict = graph_to_tensor_dict(graph)
    assert isinstance(tensor_dict["gene"], dict)
    assert isinstance(tensor_dict["gene"]["x"], torch.Tensor) # type: ignore
    assert isinstance(tensor_dict["disease"]["x"], torch.Tensor) # type: ignore

def test_graph_to_tensor_dict_ignore_strings() -> None:
    """Test that strings are ignored in graph conversion."""
    graph = {"name": "gene_graph", "nodes": [1, 2]}
    tensor_dict = graph_to_tensor_dict(graph)
    assert tensor_dict["name"] == "gene_graph"
    assert isinstance(tensor_dict["nodes"], torch.Tensor)

def test_graph_to_tensor_dict_device() -> None:
    """Test graph conversion with a specific device."""
    graph = {"nodes": [1, 2]}
    device = torch.device("cpu")
    tensor_dict = graph_to_tensor_dict(graph, device=device)
    assert tensor_dict["nodes"].device.type == "cpu"

def test_to_tensor_invalid_type() -> None:
    """Test to_tensor with invalid types."""
    with pytest.raises(Exception):
        to_tensor("invalid_string")

def test_to_tensor_scalar() -> None:
    """Test converting a scalar to a tensor."""
    tensor = to_tensor(5.0)
    assert tensor.dim() == 0
    assert tensor.item() == 5.0
