"""Utility functions for multi-modal operations."""
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

def pad_to_match(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Pad a list of numpy arrays to have the same shape.

    Assumes that the arrays differ only in the first dimension, and pads
    with zeros at the end of the first dimension to match the largest one.

    Args:
        arrays: A list of numpy arrays to pad.

    Returns:
        A list of padded numpy arrays all having the same shape.
    """
    if not arrays:
        return []
    
    max_len = max(arr.shape[0] for arr in arrays)
    padded_arrays = []
    
    for arr in arrays:
        pad_width = [(0, max_len - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
        padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded_arr)
        
    return padded_arrays

def concatenate_dict(arrays_dict: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate lists of arrays in a dictionary along the first axis.

    Args:
        arrays_dict: A dictionary mapping strings to lists of numpy arrays.

    Returns:
        A dictionary mapping strings to concatenated numpy arrays.
    """
    concatenated = {}
    for key, arr_list in arrays_dict.items():
        if arr_list:
            concatenated[key] = np.concatenate(arr_list, axis=0)
        else:
            concatenated[key] = np.array([])
    return concatenated

def mask_array(array: np.ndarray, mask_val: float) -> np.ndarray:
    """Create a boolean mask for an array where elements equal mask_val.

    Args:
        array: The numpy array to mask.
        mask_val: The value to mask.

    Returns:
        A boolean numpy array where True indicates the element is not mask_val.
    """
    if np.isnan(mask_val):
        return ~np.isnan(array)
    return array != mask_val

def check_nan(array: np.ndarray) -> bool:
    """Check if a numpy array contains any NaN values.

    Args:
        array: The numpy array to check.

    Returns:
        True if the array contains NaN values, False otherwise.
    """
    return np.isnan(array).any()
