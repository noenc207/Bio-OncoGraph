"""Utility functions for foundation models."""

import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with epsilon.

    Args:
        x: Input array of shape (N, D).

    Returns:
        L2 normalized array.
    """
    epsilon = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + epsilon)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Stable cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity scalar.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.clip(np.dot(a_norm, b_norm), -1.0, 1.0))

def pairwise_cosine_similarity(x: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix for a set of vectors.

    Args:
        x: Input array of shape (N, D).

    Returns:
        NxN similarity matrix.
    """
    x_norm = l2_normalize(x)
    return np.clip(np.dot(x_norm, x_norm.T), -1.0, 1.0)

def compute_alignment(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """Compute mean pairwise alignment (expected distance between positive pairs).
    
    Expected distance is typically the L2 distance between normalized embeddings.

    Args:
        embeddings1: First set of embeddings (N, D).
        embeddings2: Second set of embeddings (N, D).

    Returns:
        Mean alignment scalar.
    """
    e1_norm = l2_normalize(embeddings1)
    e2_norm = l2_normalize(embeddings2)
    distances = np.linalg.norm(e1_norm - e2_norm, axis=1)
    return float(np.mean(distances**2))

def compute_uniformity(embeddings: np.ndarray) -> float:
    """Compute uniformity of embeddings (log of mean pairwise Gaussian kernel).
    
    A lower value indicates a more uniform distribution.

    Args:
        embeddings: Input embeddings (N, D).

    Returns:
        Uniformity scalar.
    """
    t = 2.0
    e_norm = l2_normalize(embeddings)
    # Compute squared distances efficiently
    sq_dists = np.sum((e_norm[:, None, :] - e_norm[None, :, :]) ** 2, axis=-1)
    # Compute Gaussian kernel
    kernel = np.exp(-t * sq_dists)
    # Mask out diagonal to avoid comparing vectors with themselves
    np.fill_diagonal(kernel, 0.0)
    
    num_elements = kernel.shape[0] * (kernel.shape[0] - 1)
    if num_elements <= 0:
        return 0.0
        
    mean_kernel = np.sum(kernel) / num_elements
    return float(np.log(mean_kernel + 1e-12))

def random_split_indices(n: int, train_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic random split of indices.

    Args:
        n: Total number of items.
        train_ratio: Proportion of items to include in the training set (0.0 to 1.0).
        seed: Random seed for determinism.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return train_indices, test_indices
