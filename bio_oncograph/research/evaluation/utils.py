"""General utilities for scientific evaluation."""

import logging
from typing import List, Tuple
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def compute_mean_std(data: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation of a dataset.

    Args:
        data: List of numerical values.

    Returns:
        A tuple of (mean, standard deviation).
    """
    logger.debug("Computing mean and standard deviation")
    if not data:
        return (0.0, 0.0)
    return (float(np.mean(data)), float(np.std(data)))


def format_p_value(p: float) -> str:
    """Format a p-value for reporting.

    Args:
        p: The p-value.

    Returns:
        A formatted string representation.
    """
    logger.debug("Formatting p-value: %f", p)
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    return f"p = {p:.3f}"


def multiple_comparison_correction(
    p_values: List[float], method: str = 'fdr'
) -> npt.NDArray[np.float64]:
    """Apply multiple comparison correction to a list of p-values.

    Args:
        p_values: List of raw p-values.
        method: Correction method ('bonferroni' or 'fdr').

    Returns:
        An array of corrected p-values.
    """
    logger.info("Applying multiple comparison correction using method: %s", method)
    p_arr = np.array(p_values)
    n = len(p_arr)
    if n == 0:
        return p_arr

    if method.lower() == 'bonferroni':
        corrected = np.minimum(p_arr * n, 1.0)
    elif method.lower() == 'fdr':
        # Benjamini-Hochberg FDR mock implementation
        order = np.argsort(p_arr)
        sorted_p = p_arr[order]
        corrected_sorted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
            
        corrected = np.zeros(n)
        corrected[order] = corrected_sorted
    else:
        logger.warning("Unknown correction method '%s', returning raw p-values", method)
        corrected = p_arr

    return corrected
