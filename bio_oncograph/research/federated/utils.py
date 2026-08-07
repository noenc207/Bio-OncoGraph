import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

def compute_weighted_average(arrays: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Computes the weighted average of a list of numpy arrays.
    
    Args:
        arrays: List of numpy arrays to average.
        weights: List of weights corresponding to each array.
        
    Returns:
        The weighted average array.
        
    Raises:
        ValueError: If arrays and weights have different lengths or are empty.
    """
    if not arrays or not weights:
        raise ValueError("Arrays and weights must not be empty.")
    if len(arrays) != len(weights):
        raise ValueError("Length of arrays and weights must match.")
        
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights must be non-zero.")
        
    weighted_sum = sum(a * w for a, w in zip(arrays, weights))
    return weighted_sum / total_weight

def add_gaussian_noise(array: np.ndarray, sensitivity: float, epsilon: float, delta: float, seed: Optional[int] = None) -> np.ndarray:
    """Adds Gaussian noise to an array for differential privacy.
    
    Args:
        array: Input numpy array.
        sensitivity: L2 sensitivity of the query.
        epsilon: Privacy budget.
        delta: Probability of failure.
        seed: Random seed for reproducibility.
        
    Returns:
        Numpy array with added Gaussian noise.
    """
    if seed is not None:
        np.random.seed(seed)
        
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=array.shape)
    return array + noise

def simulate_network_delay(base_delay: float, jitter: float, seed: Optional[int] = None) -> float:
    """Simulates network delay with jitter.
    
    Args:
        base_delay: Base delay in seconds.
        jitter: Maximum variation from base delay in seconds.
        seed: Random seed for reproducibility.
        
    Returns:
        Simulated network delay in seconds.
    """
    if seed is not None:
        np.random.seed(seed)
        
    delay = base_delay + np.random.uniform(-jitter, jitter)
    return max(0.0, delay)
