"""Input validation for image preprocessing."""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("BioOncoGraph")

_SUPPORTED_DTYPES = (np.uint8, np.float32, np.float64)


def validate_preprocessing_input(
    image: np.ndarray,
    min_resolution: Tuple[int, int] = (16, 16),
    max_resolution: Tuple[int, int] = (4096, 4096),
) -> None:
    """Run all validation checks. Raises on first failure."""
    _validate_type(image)
    _validate_not_empty(image)
    _validate_dtype(image)
    _validate_dimensions(image)
    _validate_channels(image)
    _validate_no_nan(image)
    _validate_no_inf(image)
    _validate_min_resolution(image, min_resolution)
    _validate_max_resolution(image, max_resolution)


def _validate_type(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}.")


def _validate_not_empty(image: np.ndarray) -> None:
    if image.size == 0:
        raise ValueError("Image array is empty (size=0).")


def _validate_dtype(image: np.ndarray) -> None:
    if image.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {image.dtype}.")


def _validate_dimensions(image: np.ndarray) -> None:
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2-D or 3-D, got ndim={image.ndim}.")


def _validate_channels(image: np.ndarray) -> None:
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Invalid channels: {image.shape[2]}.")


def _validate_no_nan(image: np.ndarray) -> None:
    if np.issubdtype(image.dtype, np.floating) and np.any(np.isnan(image)):
        raise ValueError("Image contains NaN values.")


def _validate_no_inf(image: np.ndarray) -> None:
    if np.issubdtype(image.dtype, np.floating) and np.any(np.isinf(image)):
        raise ValueError("Image contains Inf values.")


def _validate_min_resolution(
    image: np.ndarray, min_res: Tuple[int, int]
) -> None:
    h, w = image.shape[0], image.shape[1]
    if h < min_res[0] or w < min_res[1]:
        raise ValueError(f"Image {h}x{w} below minimum {min_res[0]}x{min_res[1]}.")


def _validate_max_resolution(
    image: np.ndarray, max_res: Tuple[int, int]
) -> None:
    h, w = image.shape[0], image.shape[1]
    if h > max_res[0] or w > max_res[1]:
        raise ValueError(f"Image {h}x{w} exceeds maximum {max_res[0]}x{max_res[1]}.")
