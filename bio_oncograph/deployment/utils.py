"""Utility functions for the deployment platform."""
import uuid
import logging
import numpy as np

logger = logging.getLogger(__name__)

def compute_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes a divergence metric between two distributions.

    Args:
        p: First distribution array.
        q: Second distribution array.

    Returns:
        The calculated divergence score.
    """
    logger.debug("Computing divergence between two arrays.")
    if p.shape != q.shape:
        return float(np.abs(np.mean(p) - np.mean(q)))
    return float(np.mean((p - q) ** 2))

def generate_request_id() -> str:
    """Generates a unique request identifier.

    Returns:
        A unique string identifier.
    """
    req_id = str(uuid.uuid4())
    logger.debug("Generated request ID: %s", req_id)
    return req_id
