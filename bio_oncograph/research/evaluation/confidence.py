import logging
import numpy as np
from typing import Tuple, List, Union

logger = logging.getLogger(__name__)

class ConfidenceEstimator:
    """Estimates confidence intervals for metrics."""
    
    def bootstrap_ci(self, data: Union[List[float], np.ndarray], num_samples: int, alpha: float) -> Tuple[float, float]:
        """Calculates bootstrap confidence interval."""
        arr = np.array(data)
        n = len(arr)
        if n == 0:
            return (0.0, 0.0)
        means = np.array([np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(num_samples)])
        lower = float(np.percentile(means, alpha / 2 * 100))
        upper = float(np.percentile(means, (1 - alpha / 2) * 100))
        return (lower, upper)

    def jackknife_ci(self, data: Union[List[float], np.ndarray], alpha: float) -> Tuple[float, float]:
        """Calculates jackknife confidence interval."""
        arr = np.array(data)
        n = len(arr)
        if n <= 1:
            return (0.0, 0.0)
        jackknife_samples = np.zeros(n)
        for i in range(n):
            jackknife_samples[i] = np.mean(np.delete(arr, i))
        
        mean_jk = np.mean(jackknife_samples)
        std_err = np.sqrt((n - 1) * np.mean((jackknife_samples - mean_jk) ** 2))
        
        z_val = 1.96
        return (float(mean_jk - z_val * std_err), float(mean_jk + z_val * std_err))

    def empirical_ci(self, data: Union[List[float], np.ndarray], alpha: float) -> Tuple[float, float]:
        """Calculates empirical confidence interval from data quantiles."""
        arr = np.array(data)
        if len(arr) == 0:
            return (0.0, 0.0)
        lower = float(np.percentile(arr, alpha / 2 * 100))
        upper = float(np.percentile(arr, (1 - alpha / 2) * 100))
        return (lower, upper)
