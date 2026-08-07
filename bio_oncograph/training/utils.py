"""Utility functions for training."""

import random
from typing import Dict, Union

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Sets the seed for PyTorch, NumPy, and Python random to ensure reproducibility.

    Args:
        seed: The random seed to set.
        deterministic: Whether to enforce deterministic algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device_info() -> Dict[str, Union[str, int]]:
    """Retrieves information about the current primary computing device.

    Returns:
        A dictionary containing device information.
    """
    if torch.cuda.is_available():
        return {
            "device_type": "cuda",
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        }
    return {
        "device_type": "cpu",
        "device_count": 0,
        "device_name": "CPU",
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Counts the trainable and non-trainable parameters in a model.

    Args:
        model: The PyTorch model.

    Returns:
        A dictionary with 'trainable', 'non_trainable', and 'total' parameter counts.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
