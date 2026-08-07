"""Utility functions for foundation models."""

import hashlib
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalize_l2(array: np.ndarray) -> np.ndarray:
    """Normalizes an array using L2 norm.

    Args:
        array: The input numpy array to normalize.

    Returns:
        The L2-normalized numpy array.
    """
    logger.debug("Normalizing array with L2 norm.")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return array / norm


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between two arrays.

    Args:
        a: First numpy array.
        b: Second numpy array.

    Returns:
        The cosine similarity.
    """
    logger.debug("Computing cosine similarity.")
    a_norm = normalize_l2(a)
    b_norm = normalize_l2(b)
    return np.dot(a_norm, b_norm.T)


def generate_cache_key(data_hash: str, model_version: str) -> str:
    """Generates a cache key based on data hash and model version.

    Args:
        data_hash: The hash of the data.
        model_version: The version of the model.

    Returns:
        A unique cache key string.
    """
    logger.debug("Generating cache key.")
    raw_key = f"{data_hash}_{model_version}".encode("utf-8")
    return hashlib.sha256(raw_key).hexdigest()
