"""Utility functions for PyTorch conversions and operations."""
import torch
import numpy as np
from typing import Any, Union


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The tensor to convert.

    Returns:
        np.ndarray: The corresponding NumPy array.
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Converts a NumPy array to a PyTorch tensor.

    Args:
        array (np.ndarray): The NumPy array to convert.
        dtype (torch.dtype): The desired tensor data type.
        device (torch.device): The device to place the tensor on.

    Returns:
        torch.Tensor: The corresponding PyTorch tensor.
    """
    return torch.from_numpy(array).to(dtype=dtype, device=device)


def get_num_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Calculates the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        trainable_only (bool): If True, counts only trainable parameters.

    Returns:
        int: The number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
