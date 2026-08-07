"""Weight initialization functions for the Bio-OncoGraph M015 Model."""

from typing import Tuple, Callable

import numpy as np


def xavier_uniform(shape: Tuple[int, ...], seed: int) -> np.ndarray:
    """Initializes a weight array using Xavier/Glorot uniform initialization.
    
    Args:
        shape: The shape of the weight array.
        seed: The random seed for reproducibility.
        
    Returns:
        The initialized weight array.
    """
    rng = np.random.default_rng(seed)
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions for Xavier initialization.")
    fan_in = shape[0]
    fan_out = shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape)


def kaiming_uniform(shape: Tuple[int, ...], seed: int) -> np.ndarray:
    """Initializes a weight array using Kaiming/He uniform initialization.
    
    Args:
        shape: The shape of the weight array.
        seed: The random seed for reproducibility.
        
    Returns:
        The initialized weight array.
    """
    rng = np.random.default_rng(seed)
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions for Kaiming initialization.")
    fan_in = shape[0]
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=shape)


def orthogonal(shape: Tuple[int, ...], seed: int) -> np.ndarray:
    """Initializes a weight array with an orthogonal matrix.
    
    Args:
        shape: The shape of the weight array. Must be 2D.
        seed: The random seed for reproducibility.
        
    Returns:
        The initialized orthogonal weight array.
    """
    rng = np.random.default_rng(seed)
    if len(shape) != 2:
        raise ValueError("Shape must be exactly 2 dimensions for orthogonal initialization.")
    
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = rng.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_initializer(method: str) -> Callable[[Tuple[int, ...], int], np.ndarray]:
    """Retrieves an initialization function by name.
    
    Args:
        method: The name of the initialization method ('xavier', 'kaiming', 'orthogonal').
        
    Returns:
        The initialization function.
        
    Raises:
        ValueError: If the specified method is not supported.
    """
    methods = {
        'xavier': xavier_uniform,
        'kaiming': kaiming_uniform,
        'orthogonal': orthogonal
    }
    
    if method not in methods:
        raise ValueError(f"Unknown initialization method: {method}. "
                         f"Supported methods are: {list(methods.keys())}")
        
    return methods[method]
