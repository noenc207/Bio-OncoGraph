"""Serialization utilities for PyG Data and Batch objects."""

import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from typing import Any, Dict, Optional, Union
import os


def save_pyg_data(data: Union[Data, Batch], filepath: str) -> None:
    """Serialize and save a PyG Data or Batch object to a file.

    Args:
        data (Union[Data, Batch]): The PyG data object to serialize.
        filepath (str): The destination file path.

    Raises:
        TypeError: If the input data is not a PyG Data or Batch object.
    """
    if not isinstance(data, (Data, Batch)):
        raise TypeError("Input must be a PyG Data or Batch object.")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    payload: Dict[str, Any] = {
        "pyg_version": torch_geometric.__version__,
        "torch_version": torch.__version__,
        "data": data.to_dict() if isinstance(data, Data) else data,
        "is_batch": isinstance(data, Batch)
    }
    
    torch.save(payload, filepath)


def load_pyg_data(filepath: str, device: Optional[torch.device] = None) -> Union[Data, Batch]:
    """Load and deserialize a PyG Data or Batch object from a file.

    Args:
        filepath (str): The source file path.
        device (Optional[torch.device], optional): The device to map the tensors to. Defaults to None.

    Returns:
        Union[Data, Batch]: The loaded PyG data object.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If there is a version mismatch or corruption.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    payload = torch.load(filepath, map_location=device)
    
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Invalid file format. The file is corrupted or not created by save_pyg_data.")

    is_batch = payload.get("is_batch", False)
    data_dict = payload["data"]

    if is_batch:
        if isinstance(data_dict, Batch):
            return data_dict.to(device) if device else data_dict
        else:
            raise RuntimeError("Corrupted Batch object in serialization payload.")
    else:
        data = Data.from_dict(data_dict)
        return data.to(device) if device else data
