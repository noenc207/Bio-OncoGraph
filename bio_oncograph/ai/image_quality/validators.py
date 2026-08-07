"""
Image validators for Bio-OncoGraph IQA.

Provides fail-fast validation functions that raise ``ValueError``
when an image array does not meet basic integrity requirements.
All functions accept ``numpy.ndarray`` inputs.

Design references:
  - Software Architecture § 22 (AI Modules — Input Validation)
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("BioOncoGraph")

_SUPPORTED_DTYPES = (np.uint8, np.float32, np.float64)
_VALID_CHANNEL_COUNTS = (1, 3, 4)


def validate_image(
    image: np.ndarray,
    min_resolution: Tuple[int, int] = (1, 1),
) -> None:
    """
    Run all image validation checks.

    Calls each individual validator in order.  The first failure
    raises ``ValueError``.

    Args:
        image: Image array to validate.
        min_resolution: ``(min_height, min_width)``.

    Raises:
        TypeError: If *image* is not an ``ndarray``.
        ValueError: If any check fails.
    """
    validate_type(image)
    validate_not_empty(image)
    validate_dtype(image)
    validate_dimensions(image)
    validate_channels(image)
    validate_no_nan(image)
    validate_resolution(image, min_resolution)


def validate_type(image: np.ndarray) -> None:
    """
    Ensure *image* is a numpy ``ndarray``.

    Raises:
        TypeError: If *image* is not an ``ndarray``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got "
            f"{type(image).__name__}."
        )


def validate_not_empty(image: np.ndarray) -> None:
    """
    Ensure the array is not empty.

    Raises:
        ValueError: If *image* has zero elements.
    """
    if image.size == 0:
        raise ValueError("Image array is empty (size=0).")


def validate_dtype(image: np.ndarray) -> None:
    """
    Ensure the dtype is supported.

    Supported: ``uint8``, ``float32``, ``float64``.

    Raises:
        ValueError: If dtype is unsupported.
    """
    if image.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported image dtype: {image.dtype}. "
            f"Supported: {[d.__name__ for d in _SUPPORTED_DTYPES]}."
        )


def validate_dimensions(image: np.ndarray) -> None:
    """
    Ensure the array is 2-D (grayscale) or 3-D (colour).

    Raises:
        ValueError: If ndim is not 2 or 3.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"Image must be 2-D or 3-D, got ndim={image.ndim}."
        )


def validate_channels(image: np.ndarray) -> None:
    """
    Ensure colour images have 1, 3, or 4 channels.

    Grayscale (2-D) arrays always pass.

    Raises:
        ValueError: If channel count is invalid.
    """
    if image.ndim == 3:
        c = image.shape[2]
        if c not in _VALID_CHANNEL_COUNTS:
            raise ValueError(
                f"Invalid number of channels: {c}. "
                f"Expected one of {_VALID_CHANNEL_COUNTS}."
            )


def validate_no_nan(image: np.ndarray) -> None:
    """
    Ensure the image contains no NaN values.

    Integer dtypes are skipped (cannot contain NaN).

    Raises:
        ValueError: If any NaN is found.
    """
    if np.issubdtype(image.dtype, np.floating):
        if np.any(np.isnan(image)):
            raise ValueError(
                "Image contains NaN values."
            )


def validate_resolution(
    image: np.ndarray,
    min_resolution: Tuple[int, int] = (1, 1),
) -> None:
    """
    Ensure the image meets a minimum resolution.

    Args:
        image: Image array.
        min_resolution: ``(min_height, min_width)``.

    Raises:
        ValueError: If the image is too small.
    """
    h, w = image.shape[0], image.shape[1]
    min_h, min_w = min_resolution
    if h < min_h or w < min_w:
        raise ValueError(
            f"Image resolution {h}×{w} is below "
            f"minimum {min_h}×{min_w}."
        )
