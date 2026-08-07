"""Visualization utilities for explainability."""

import numpy as np


def format_explanation_for_visualization(attribution_scores: np.ndarray) -> np.ndarray:
    """Normalizes and formats attribution scores for UI rendering.

    Scales the attribution scores to the range [0, 1].

    Args:
        attribution_scores: Array of attribution scores.

    Returns:
        Normalized array in the range [0, 1] suitable for visualization
        (e.g., as a heatmap).
    """
    min_val = np.min(attribution_scores)
    max_val = np.max(attribution_scores)
    
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(attribution_scores)
        
    normalized = (attribution_scores - min_val) / range_val
    return normalized
