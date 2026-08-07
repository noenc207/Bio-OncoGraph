"""Image normalisation functions.

All functions accept float64 arrays in [0, 255] and return
normalised float64 arrays.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger("BioOncoGraph")


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].

    Args:
        image: Float64 array.

    Returns:
        Array normalised to [0, 1].
    """
    img = image.astype(np.float64)
    vmin = img.min()
    vmax = img.max()
    denom = vmax - vmin
    if denom < 1e-12:
        return np.zeros_like(img)
    return (img - vmin) / denom


def normalize_meanstd(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Mean/std normalise (ImageNet-style).

    Input is assumed to be in [0, 1] range. If the input is in
    [0, 255], it is first divided by 255.

    Args:
        image: Float64 array.
        mean: Per-channel means.
        std: Per-channel standard deviations.

    Returns:
        Normalised array.
    """
    img = image.astype(np.float64)
    if img.max() > 1.0 + 1e-6:
        img = img / 255.0

    if img.ndim == 2:
        # Grayscale: use first channel's mean/std
        return (img - mean[0]) / max(std[0], 1e-12)

    c = img.shape[2]
    result = np.empty_like(img)
    for ch in range(min(c, len(mean))):
        result[:, :, ch] = (img[:, :, ch] - mean[ch]) / max(std[ch], 1e-12)
    # Copy remaining channels unchanged
    for ch in range(len(mean), c):
        result[:, :, ch] = img[:, :, ch]
    return result


def normalize_image(
    image: np.ndarray,
    mode: str = "minmax",
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Dispatch normalisation by mode.

    Args:
        image: Input array.
        mode: ``minmax``, ``meanstd``, or ``none``.
        mean: Per-channel means (for ``meanstd``).
        std: Per-channel stds (for ``meanstd``).

    Returns:
        Normalised array.

    Raises:
        ValueError: If *mode* is unknown.
    """
    if mode == "minmax":
        return normalize_minmax(image)
    if mode == "meanstd":
        return normalize_meanstd(image, mean, std)
    if mode == "none":
        return image.astype(np.float64)
    raise ValueError(f"Unknown normalization mode: {mode!r}.")
