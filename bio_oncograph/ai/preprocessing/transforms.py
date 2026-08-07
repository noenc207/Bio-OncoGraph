"""
Classical image transforms for Bio-OncoGraph preprocessing.

All functions are **deterministic**, operate on ``numpy.ndarray``
inputs (float64, range [0, 255]), and return ``numpy.ndarray``.

Categories:
  - **Geometric**: resize, center_crop, pad_to_size.
  - **Colour**: rgb_to_bgr, bgr_to_rgb, rgb_to_grayscale.
  - **Enhancement**: histogram_equalize, clahe, gamma_correction.
  - **Filtering**: gaussian_blur, median_filter, bilateral_filter,
    unsharp_mask.

**No deep-learning models are used.**

Design references:
  - Software Architecture § 22 (AI Modules — Preprocessing)
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import (  # type: ignore[import-untyped]
    gaussian_filter,
    median_filter as _scipy_median,
    zoom,
)

logger = logging.getLogger("BioOncoGraph")

# ── Internal helpers ────────────────────────────────────────────────


def _to_float64(image: np.ndarray) -> np.ndarray:
    """Convert to float64 in [0, 255] range."""
    img = image.astype(np.float64)
    if np.issubdtype(image.dtype, np.floating) and img.max() <= 1.0 + 1e-6:
        img = img * 255.0
    return img


def _ensure_3d(image: np.ndarray) -> np.ndarray:
    """Ensure image is (H, W, C).  Grayscale → (H, W, 1)."""
    if image.ndim == 2:
        return image[:, :, np.newaxis]
    return image


# ═══════════════════════════════════════════════════════════════════
# Geometric transforms
# ═══════════════════════════════════════════════════════════════════


def resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize an image to ``target_size``.

    Args:
        image: Input array (2-D or 3-D).
        target_size: ``(height, width)``.
        interpolation: ``nearest`` (0), ``bilinear`` (1),
            or ``bicubic`` (3).

    Returns:
        Resized array.
    """
    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    order = order_map.get(interpolation, 1)

    h, w = image.shape[0], image.shape[1]
    th, tw = target_size
    zh = th / max(h, 1)
    zw = tw / max(w, 1)

    img = image.astype(np.float64)
    if img.ndim == 3:
        factors: Tuple[float, ...] = (zh, zw, 1.0)
    else:
        factors = (zh, zw)

    return zoom(img, factors, order=order, mode="reflect")


def center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int],
) -> np.ndarray:
    """
    Centre-crop an image.

    Args:
        image: Input array.
        crop_size: ``(height, width)`` of the crop.

    Returns:
        Cropped array.

    Raises:
        ValueError: If *crop_size* exceeds image dimensions.
    """
    h, w = image.shape[0], image.shape[1]
    ch, cw = crop_size
    if ch > h or cw > w:
        raise ValueError(
            f"Crop size ({ch}, {cw}) exceeds "
            f"image size ({h}, {w})."
        )
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0: y0 + ch, x0: x0 + cw].copy()


def pad_to_size(
    image: np.ndarray,
    target_size: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """
    Pad an image to at least ``target_size``.

    Args:
        image: Input array.
        target_size: Minimum ``(height, width)``.
        mode: ``constant``, ``reflect``, or ``edge``.
        value: Fill value for ``constant`` mode.

    Returns:
        Padded array (or original if already large enough).
    """
    h, w = image.shape[0], image.shape[1]
    th, tw = target_size
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return image.copy()

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if image.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))

    kwargs = {}
    if mode == "constant":
        kwargs["constant_values"] = value

    return np.pad(
        image.astype(np.float64), pad_width, mode=mode, **kwargs
    )


# ═══════════════════════════════════════════════════════════════════
# Colour transforms
# ═══════════════════════════════════════════════════════════════════


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB to BGR by reversing channel order.

    Args:
        image: 3-D array with ≥ 3 channels.

    Returns:
        BGR array.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("rgb_to_bgr requires a 3-channel image.")
    return image[:, :, ::-1].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB (same as ``rgb_to_bgr``)."""
    return rgb_to_bgr(image)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert to single-channel grayscale using ITU-R BT.601.

    Args:
        image: 3-D array with ≥ 3 channels.

    Returns:
        2-D float64 array.
    """
    img = _to_float64(image)
    if img.ndim == 2:
        return img
    return (
        0.2989 * img[:, :, 0]
        + 0.5870 * img[:, :, 1]
        + 0.1140 * img[:, :, 2]
    )


# ═══════════════════════════════════════════════════════════════════
# Enhancement transforms
# ═══════════════════════════════════════════════════════════════════


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalisation on each channel independently.

    Args:
        image: Input array (float64, [0, 255]).

    Returns:
        Equalised array in [0, 255].
    """
    img = _to_float64(image)

    def _eq_channel(ch: np.ndarray) -> np.ndarray:
        vals = np.clip(ch, 0, 255).astype(np.uint8)
        hist, _ = np.histogram(vals.ravel(), bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float64)
        cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
        denom = max(float(ch.size) - cdf_min, 1.0)
        lut = (cdf - cdf_min) / denom * 255.0
        return lut[vals]

    if img.ndim == 2:
        return _eq_channel(img)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _eq_channel(img[:, :, c])
    return result


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation.

    Vectorised implementation with bilinear interpolation
    between tile CDFs for smooth output.

    Args:
        image: Input (float64, [0, 255]).
        clip_limit: Histogram clipping factor.
        grid_size: ``(rows, cols)`` tile grid.

    Returns:
        CLAHE-enhanced array in [0, 255].
    """
    img = _to_float64(image)

    def _clahe_single(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        gh, gw = grid_size
        tile_h = h / gh
        tile_w = w / gw

        # Compute clipped CDF per tile
        cdfs = np.zeros((gh, gw, 256), dtype=np.float64)
        for ti in range(gh):
            for tj in range(gw):
                y1 = int(ti * tile_h)
                y2 = int((ti + 1) * tile_h) if ti < gh - 1 else h
                x1 = int(tj * tile_w)
                x2 = int((tj + 1) * tile_w) if tj < gw - 1 else w
                cdfs[ti, tj] = _clipped_cdf(
                    gray[y1:y2, x1:x2], clip_limit
                )

        # Bilinear interpolation (vectorised)
        yy, xx = np.mgrid[0:h, 0:w]
        ty = np.clip((yy + 0.5) / tile_h - 0.5, 0, gh - 1.001)
        tx = np.clip((xx + 0.5) / tile_w - 0.5, 0, gw - 1.001)

        ti0 = np.clip(ty.astype(np.intp), 0, gh - 2)
        tj0 = np.clip(tx.astype(np.intp), 0, gw - 2)
        ti1 = ti0 + 1
        tj1 = tj0 + 1

        dy = np.clip(ty - ti0, 0.0, 1.0)
        dx = np.clip(tx - tj0, 0.0, 1.0)

        vals = np.clip(gray, 0, 255).astype(np.intp)

        v00 = cdfs[ti0, tj0, vals]
        v01 = cdfs[ti0, tj1, vals]
        v10 = cdfs[ti1, tj0, vals]
        v11 = cdfs[ti1, tj1, vals]

        return np.clip(
            v00 * (1 - dy) * (1 - dx)
            + v01 * (1 - dy) * dx
            + v10 * dy * (1 - dx)
            + v11 * dy * dx,
            0, 255,
        )

    if img.ndim == 2:
        return _clahe_single(img)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[:, :, c] = _clahe_single(img[:, :, c])
    return result


def _clipped_cdf(
    tile: np.ndarray, clip_limit: float
) -> np.ndarray:
    """Compute a contrast-limited CDF for a single tile."""
    vals = np.clip(tile, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(vals.ravel(), bins=256, range=(0, 256))
    limit = max(1, int(clip_limit * tile.size / 256))
    excess = int(np.sum(np.maximum(hist - limit, 0)))
    hist = np.minimum(hist, limit)
    hist += excess // 256

    cdf = hist.cumsum().astype(np.float64)
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0.0
    denom = max(float(tile.size) - cdf_min, 1.0)
    return (cdf - cdf_min) / denom * 255.0


def gamma_correction(
    image: np.ndarray, gamma: float = 1.0
) -> np.ndarray:
    """
    Apply gamma correction: ``out = 255 * (in / 255) ** gamma``.

    Args:
        image: Input (float64, [0, 255]).
        gamma: Exponent (< 1 brightens, > 1 darkens).

    Returns:
        Gamma-corrected array in [0, 255].
    """
    img = np.clip(_to_float64(image), 0, 255) / 255.0
    return np.clip(np.power(img, gamma) * 255.0, 0, 255)


# ═══════════════════════════════════════════════════════════════════
# Filtering transforms
# ═══════════════════════════════════════════════════════════════════


def gaussian_blur(
    image: np.ndarray, sigma: float = 1.0
) -> np.ndarray:
    """
    Gaussian blur via ``scipy.ndimage.gaussian_filter``.

    Args:
        image: Input array.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Blurred array.
    """
    img = _to_float64(image)
    if img.ndim == 3:
        return gaussian_filter(img, sigma=(sigma, sigma, 0))
    return gaussian_filter(img, sigma=sigma)


def median_filter_op(
    image: np.ndarray, size: int = 3
) -> np.ndarray:
    """
    Median filter via ``scipy.ndimage.median_filter``.

    Args:
        image: Input array.
        size: Kernel size.

    Returns:
        Filtered array.
    """
    img = _to_float64(image)
    if img.ndim == 3:
        return _scipy_median(img, size=(size, size, 1))
    return _scipy_median(img, size=size)


def bilateral_filter(
    image: np.ndarray,
    spatial_sigma: float = 2.0,
    range_sigma: float = 25.0,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Bilateral filter (vectorised over pixels).

    Combines spatial proximity and intensity similarity
    weighting to smooth while preserving edges.

    Args:
        image: 2-D grayscale or 3-D colour array.
        spatial_sigma: Spatial Gaussian sigma.
        range_sigma: Intensity Gaussian sigma.
        kernel_size: Kernel side length (odd).

    Returns:
        Filtered array.
    """
    img = _to_float64(image)

    def _bilateral_2d(gray: np.ndarray) -> np.ndarray:
        half = kernel_size // 2
        padded = np.pad(gray, half, mode="reflect")
        y_k, x_k = np.ogrid[-half: half + 1, -half: half + 1]
        spatial = np.exp(
            -(x_k.astype(np.float64) ** 2 + y_k.astype(np.float64) ** 2)
            / (2.0 * spatial_sigma ** 2)
        )

        result = np.zeros_like(gray)
        weight_sum = np.zeros_like(gray)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                shifted = padded[
                    half + dy: half + dy + gray.shape[0],
                    half + dx: half + dx + gray.shape[1],
                ]
                range_w = np.exp(
                    -(shifted - gray) ** 2 / (2.0 * range_sigma ** 2)
                )
                w = spatial[dy + half, dx + half] * range_w
                result += shifted * w
                weight_sum += w

        return result / np.maximum(weight_sum, 1e-12)

    if img.ndim == 2:
        return _bilateral_2d(img)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _bilateral_2d(img[:, :, c])
    return out


def unsharp_mask(
    image: np.ndarray,
    strength: float = 1.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Unsharp-mask sharpening.

    ``sharpened = image + strength * (image − blurred)``

    Args:
        image: Input array.
        strength: Sharpening intensity.
        sigma: Gaussian sigma for the blur.

    Returns:
        Sharpened array in [0, 255].
    """
    img = _to_float64(image)
    blurred = gaussian_blur(img, sigma=sigma)
    sharpened = img + strength * (img - blurred)
    return np.clip(sharpened, 0, 255)
