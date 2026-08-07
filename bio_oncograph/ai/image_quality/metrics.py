"""
Classical image quality metrics for Bio-OncoGraph IQA.

All functions are **deterministic**, operate on ``numpy.ndarray``
inputs, and return normalised ``[0, 1]`` scores where ``1.0`` is
best quality.

Algorithms implemented:

  - **Blur**: Variance of Laplacian (Pech-Pacheco et al., 2000).
  - **Focus**: Mean gradient magnitude via Sobel operators.
  - **Brightness**: Mean intensity relative to an ideal range.
  - **Contrast**: Normalised intensity standard deviation.
  - **Noise**: Median Absolute Deviation of Laplacian
    (Immerkaer, 1996).
  - **Exposure**: Histogram utilisation ratio.
  - **Resolution**: Dimension adequacy vs. minimum requirement.
  - **Artifact**: Ratio of saturated / dead pixel regions.

**No deep-learning models are used.**

Design references:
  - Software Architecture § 22 (AI Modules — IQA)
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import convolve  # type: ignore[import-untyped]

logger = logging.getLogger("BioOncoGraph")

# ── Kernels ─────────────────────────────────────────────────────────

_LAPLACIAN_KERNEL = np.array(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]],
    dtype=np.float64,
)

_SOBEL_X = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float64,
)

_SOBEL_Y = np.array(
    [[-1, -2, -1],
     [0,  0,  0],
     [1,  2,  1]],
    dtype=np.float64,
)


# ── Helpers ─────────────────────────────────────────────────────────


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to single-channel float64 grayscale.

    Handles 2-D grayscale, 3-channel RGB, 4-channel RGBA,
    and single-channel ``(H, W, 1)`` inputs.  Integer images
    are normalised to ``[0, 255]`` float range.

    Args:
        image: Input image array.

    Returns:
        2-D float64 array in ``[0, 255]`` range.
    """
    img = image.astype(np.float64)

    # Normalise float images stored in [0, 1] to [0, 255]
    if np.issubdtype(image.dtype, np.floating) and img.max() <= 1.0:
        img = img * 255.0

    if img.ndim == 2:
        return img

    c = img.shape[2]
    if c == 1:
        return img[:, :, 0]
    if c >= 3:
        # ITU-R BT.601 luma coefficients
        return (
            0.2989 * img[:, :, 0]
            + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2]
        )
    raise ValueError(f"Cannot convert to grayscale: shape={image.shape}")


# ── Metric functions ───────────────────────────────────────────────


def compute_blur_score(
    image: np.ndarray,
    threshold: float = 100.0,
) -> float:
    """
    Compute blur score via Variance of Laplacian.

    Higher variance indicates sharper edges.  The raw variance is
    mapped to ``[0, 1]`` by dividing by *threshold* and clamping.

    Args:
        image: Input image.
        threshold: Variance value that maps to score ``1.0``.

    Returns:
        Blur score in ``[0, 1]``.
    """
    gray = to_grayscale(image)
    laplacian = convolve(gray, _LAPLACIAN_KERNEL, mode="reflect")
    raw = float(np.var(laplacian))
    return min(1.0, raw / max(threshold, 1e-12))


def compute_focus_score(
    image: np.ndarray,
    threshold: float = 30.0,
) -> float:
    """
    Compute focus score via mean gradient magnitude (Sobel).

    Args:
        image: Input image.
        threshold: Gradient mean that maps to score ``1.0``.

    Returns:
        Focus score in ``[0, 1]``.
    """
    gray = to_grayscale(image)
    gx = convolve(gray, _SOBEL_X, mode="reflect")
    gy = convolve(gray, _SOBEL_Y, mode="reflect")
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    raw = float(np.mean(magnitude))
    return min(1.0, raw / max(threshold, 1e-12))


def compute_brightness_score(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.1, 0.9),
) -> float:
    """
    Compute brightness appropriateness score.

    The mean intensity (normalised to ``[0, 1]``) is compared
    against the ideal *brightness_range*.  Values inside the range
    score ``1.0``; values outside decay linearly towards ``0.0``.

    Args:
        image: Input image.
        brightness_range: ``(low, high)`` ideal range.

    Returns:
        Brightness score in ``[0, 1]``.
    """
    gray = to_grayscale(image)
    mean_b = float(np.mean(gray)) / 255.0
    low, high = brightness_range

    if low <= mean_b <= high:
        return 1.0
    if mean_b < low:
        return max(0.0, mean_b / max(low, 1e-12))
    # mean_b > high
    denom = max(1.0 - high, 1e-12)
    return max(0.0, (1.0 - mean_b) / denom)


def compute_contrast_score(
    image: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Compute contrast score via normalised standard deviation.

    Args:
        image: Input image.
        threshold: Std-dev (normalised) that maps to ``1.0``.

    Returns:
        Contrast score in ``[0, 1]``.
    """
    gray = to_grayscale(image)
    std = float(np.std(gray)) / 255.0
    return min(1.0, std / max(threshold, 1e-12))


def compute_noise_score(
    image: np.ndarray,
    threshold: float = 50.0,
) -> float:
    """
    Estimate noise level and return an inverse score.

    Uses the Median Absolute Deviation of the Laplacian
    (Immerkaer, 1996) to estimate Gaussian noise sigma.

    Args:
        image: Input image.
        threshold: Sigma value at which score reaches ``0.0``.

    Returns:
        Noise score in ``[0, 1]`` (higher = less noisy).
    """
    gray = to_grayscale(image)
    laplacian = convolve(gray, _LAPLACIAN_KERNEL, mode="reflect")
    # Robust noise estimator: sigma ≈ MAD / 0.6745
    mad = float(np.median(np.abs(laplacian - np.median(laplacian))))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    return max(0.0, 1.0 - sigma / max(threshold, 1e-12))


def compute_exposure_score(image: np.ndarray) -> float:
    """
    Compute exposure score via histogram utilisation.

    A well-exposed image uses a large fraction of the dynamic
    range.  The score is the ratio of non-empty histogram bins
    to total bins, penalised for extreme clipping.

    Args:
        image: Input image.

    Returns:
        Exposure score in ``[0, 1]``.
    """
    gray = to_grayscale(image)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 255))

    # Fraction of non-empty bins
    utilisation = float(np.count_nonzero(hist)) / 256.0

    # Clipping penalty: fraction of pixels at 0 or 255
    total = float(gray.size)
    clip_low = float(hist[0]) / total
    clip_high = float(hist[255]) / total
    clip_penalty = min(1.0, (clip_low + clip_high) * 5.0)

    return max(0.0, utilisation * (1.0 - clip_penalty))


def compute_resolution_score(
    image: np.ndarray,
    minimum_resolution: Tuple[int, int] = (32, 32),
) -> float:
    """
    Compute resolution adequacy score.

    Score is ``1.0`` when both dimensions meet or exceed the
    minimum, and degrades linearly below.

    Args:
        image: Input image.
        minimum_resolution: ``(min_height, min_width)``.

    Returns:
        Resolution score in ``[0, 1]``.
    """
    h, w = image.shape[0], image.shape[1]
    min_h, min_w = minimum_resolution
    h_ratio = h / max(min_h, 1)
    w_ratio = w / max(min_w, 1)
    return min(1.0, min(h_ratio, w_ratio))


def compute_artifact_score(image: np.ndarray) -> float:
    """
    Detect common artifacts (saturated / dead regions).

    The score is ``1.0`` minus the fraction of pixels that are
    fully saturated (≥ 254) or fully dead (≤ 1).

    Args:
        image: Input image.

    Returns:
        Artifact score in ``[0, 1]`` (higher = fewer artifacts).
    """
    gray = to_grayscale(image)
    total = float(gray.size)
    saturated = float(np.sum(gray >= 254.0)) / total
    dead = float(np.sum(gray <= 1.0)) / total
    artifact_ratio = saturated + dead
    return max(0.0, 1.0 - artifact_ratio)
