"""
Unit tests for image transforms.

Covers: resize, center_crop, pad_to_size, rgb_to_bgr, bgr_to_rgb,
rgb_to_grayscale, histogram_equalize, clahe, gamma_correction,
gaussian_blur, median_filter_op, bilateral_filter, unsharp_mask.
"""

import numpy as np
import pytest

from bio_oncograph.ai.preprocessing.transforms import (
    bilateral_filter,
    center_crop,
    clahe,
    gamma_correction,
    gaussian_blur,
    histogram_equalize,
    median_filter_op,
    pad_to_size,
    resize,
    rgb_to_bgr,
    rgb_to_grayscale,
    unsharp_mask,
)

_RNG = np.random.RandomState(42)


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return _RNG.randint(20, 230, (h, w, 3), dtype=np.uint8).astype(
        np.float64
    )


# ═══════════════════════════════════════════════════════════════════
# Geometric
# ═══════════════════════════════════════════════════════════════════


def test_resize_shape() -> None:
    """Resize changes output dimensions."""
    out = resize(_rgb(64, 64), (128, 128))
    assert out.shape[0] == 128
    assert out.shape[1] == 128


def test_resize_nearest() -> None:
    """Nearest-neighbour resize works."""
    out = resize(_rgb(32, 32), (64, 64), "nearest")
    assert out.shape[:2] == (64, 64)


def test_resize_bicubic() -> None:
    """Bicubic resize works."""
    out = resize(_rgb(32, 32), (16, 16), "bicubic")
    assert out.shape[:2] == (16, 16)


def test_resize_preserves_channels() -> None:
    """Channels are preserved after resize."""
    out = resize(_rgb(32, 32), (64, 64))
    assert out.shape[2] == 3


def test_resize_grayscale() -> None:
    """2-D images can be resized."""
    gray = np.zeros((32, 32), dtype=np.float64)
    out = resize(gray, (64, 64))
    assert out.shape == (64, 64)


def test_center_crop_shape() -> None:
    """Center crop returns correct size."""
    out = center_crop(_rgb(100, 100), (50, 50))
    assert out.shape[:2] == (50, 50)


def test_center_crop_too_large_raises() -> None:
    """Crop larger than image raises ValueError."""
    with pytest.raises(ValueError, match="exceeds"):
        center_crop(_rgb(32, 32), (64, 64))


def test_pad_to_size_adds_padding() -> None:
    """pad_to_size increases dimensions."""
    out = pad_to_size(_rgb(32, 32), (64, 64))
    assert out.shape[0] == 64
    assert out.shape[1] == 64


def test_pad_to_size_no_change() -> None:
    """Large-enough image is returned unchanged."""
    img = _rgb(64, 64)
    out = pad_to_size(img, (32, 32))
    assert out.shape[:2] == (64, 64)


def test_pad_reflect() -> None:
    """Reflect padding mode works."""
    out = pad_to_size(_rgb(32, 32), (64, 64), mode="reflect")
    assert out.shape[:2] == (64, 64)


# ═══════════════════════════════════════════════════════════════════
# Colour
# ═══════════════════════════════════════════════════════════════════


def test_rgb_to_bgr() -> None:
    """Channel order is reversed."""
    img = np.zeros((4, 4, 3), dtype=np.float64)
    img[:, :, 0] = 100  # R
    img[:, :, 2] = 200  # B
    bgr = rgb_to_bgr(img)
    assert bgr[0, 0, 0] == 200  # was B, now first
    assert bgr[0, 0, 2] == 100  # was R, now last


def test_rgb_to_grayscale_shape() -> None:
    """Grayscale output is 2-D."""
    out = rgb_to_grayscale(_rgb(32, 32))
    assert out.ndim == 2


def test_rgb_to_grayscale_already_gray() -> None:
    """2-D input passes through."""
    gray = np.zeros((32, 32), dtype=np.float64)
    out = rgb_to_grayscale(gray)
    assert out.ndim == 2


# ═══════════════════════════════════════════════════════════════════
# Enhancement
# ═══════════════════════════════════════════════════════════════════


def test_histogram_equalize_range() -> None:
    """Output stays in [0, 255]."""
    out = histogram_equalize(_rgb(64, 64))
    assert out.min() >= 0
    assert out.max() <= 255.1


def test_histogram_equalize_grayscale() -> None:
    """Works on 2-D images."""
    gray = _RNG.randint(50, 200, (64, 64)).astype(np.float64)
    out = histogram_equalize(gray)
    assert out.ndim == 2


def test_clahe_range() -> None:
    """CLAHE output is in [0, 255]."""
    out = clahe(_rgb(64, 64), clip_limit=2.0, grid_size=(4, 4))
    assert out.min() >= -0.5
    assert out.max() <= 255.5


def test_clahe_grayscale() -> None:
    """CLAHE works on 2-D images."""
    gray = _RNG.randint(50, 200, (64, 64)).astype(np.float64)
    out = clahe(gray, clip_limit=2.0, grid_size=(4, 4))
    assert out.ndim == 2


def test_gamma_brightens() -> None:
    """Gamma < 1 brightens the image."""
    img = np.full((32, 32, 3), 128.0)
    bright = gamma_correction(img, gamma=0.5)
    assert bright.mean() > img.mean()


def test_gamma_darkens() -> None:
    """Gamma > 1 darkens the image."""
    img = np.full((32, 32, 3), 128.0)
    dark = gamma_correction(img, gamma=2.0)
    assert dark.mean() < img.mean()


def test_gamma_identity() -> None:
    """Gamma 1.0 preserves values."""
    img = np.full((32, 32, 3), 100.0)
    out = gamma_correction(img, gamma=1.0)
    np.testing.assert_allclose(out, img, atol=0.5)


# ═══════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════


def test_gaussian_blur_smooths() -> None:
    """Blurred image has lower variance."""
    img = _rgb(64, 64)
    blurred = gaussian_blur(img, sigma=3.0)
    assert np.var(blurred) < np.var(img)


def test_gaussian_blur_preserves_shape() -> None:
    """Shape is preserved."""
    img = _rgb(64, 64)
    assert gaussian_blur(img, sigma=1.0).shape == img.shape


def test_median_filter_shape() -> None:
    """Shape is preserved."""
    img = _rgb(64, 64)
    assert median_filter_op(img, size=3).shape == img.shape


def test_median_filter_removes_speckle() -> None:
    """Median filter reduces salt-and-pepper noise."""
    img = np.full((64, 64), 128.0)
    img[10, 10] = 0.0
    img[20, 20] = 255.0
    out = median_filter_op(img, size=3)
    assert abs(out[10, 10] - 128.0) < 1.0


def test_bilateral_shape() -> None:
    """Shape is preserved."""
    img = _rgb(32, 32)
    out = bilateral_filter(img, kernel_size=3)
    assert out.shape == img.shape


def test_bilateral_grayscale() -> None:
    """Works on 2-D."""
    gray = np.full((32, 32), 100.0)
    out = bilateral_filter(gray, kernel_size=3)
    assert out.ndim == 2


def test_unsharp_mask_sharpens() -> None:
    """Sharpened image has higher gradient energy."""
    img = _rgb(64, 64)
    sharp = unsharp_mask(img, strength=2.0, sigma=1.0)
    grad_orig = np.diff(img, axis=0).var()
    grad_sharp = np.diff(sharp, axis=0).var()
    assert grad_sharp >= grad_orig


def test_unsharp_mask_range() -> None:
    """Output clipped to [0, 255]."""
    out = unsharp_mask(_rgb(64, 64), strength=5.0)
    assert out.min() >= 0
    assert out.max() <= 255.1


# ═══════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════


def test_resize_deterministic() -> None:
    """Same input → same output."""
    img = _rgb(32, 32)
    a = resize(img, (64, 64))
    b = resize(img, (64, 64))
    np.testing.assert_array_equal(a, b)


def test_clahe_deterministic() -> None:
    """Same input → same output."""
    img = _rgb(64, 64)
    a = clahe(img, grid_size=(4, 4))
    b = clahe(img, grid_size=(4, 4))
    np.testing.assert_array_equal(a, b)
