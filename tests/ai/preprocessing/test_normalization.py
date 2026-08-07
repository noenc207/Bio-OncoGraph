"""Tests for image normalization functions."""

import numpy as np
import pytest

from bio_oncograph.ai.preprocessing.normalization import (
    normalize_minmax,
    normalize_meanstd,
    normalize_image,
)


def test_minmax_basic() -> None:
    """Test minmax normalization with a uniform image."""
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = normalize_minmax(image)
    assert np.all(result == 0.0)


def test_minmax_range() -> None:
    """Test minmax normalization maps to [0, 1]."""
    image = np.array([[[0], [128], [255]]], dtype=np.uint8)
    result = normalize_minmax(image)
    assert np.isclose(result[0, 0, 0], 0.0)
    assert np.isclose(result[0, 1, 0], 128.0 / 255.0)
    assert np.isclose(result[0, 2, 0], 1.0)


def test_meanstd_basic() -> None:
    """Test mean/std normalization normalizes per channel."""
    image = np.array([[[100, 200, 50], [150, 250, 100]]], dtype=np.float32)
    result = normalize_meanstd(image)
    assert np.allclose(np.mean(result, axis=(0, 1)), 0.0, atol=1e-5)
    assert np.allclose(np.std(result, axis=(0, 1)), 1.0, atol=1e-5)


def test_meanstd_grayscale() -> None:
    """Test mean/std normalization works for grayscale images."""
    image = np.array([[100, 150], [200, 250]], dtype=np.float32)
    result = normalize_meanstd(image)
    assert np.isclose(np.mean(result), 0.0, atol=1e-5)
    assert np.isclose(np.std(result), 1.0, atol=1e-5)


def test_normalize_dispatch_minmax() -> None:
    """Test normalize_image dispatches to minmax correctly."""
    image = np.array([[[0], [128], [255]]], dtype=np.uint8)
    result = normalize_image(image, mode='minmax')
    assert np.isclose(result[0, 0, 0], 0.0)
    assert np.isclose(result[0, 2, 0], 1.0)


def test_normalize_dispatch_meanstd() -> None:
    """Test normalize_image dispatches to meanstd correctly."""
    image = np.array([[100, 150], [200, 250]], dtype=np.float32)
    result = normalize_image(image, mode='meanstd')
    assert np.isclose(np.mean(result), 0.0, atol=1e-5)
    assert np.isclose(np.std(result), 1.0, atol=1e-5)


def test_normalize_dispatch_none() -> None:
    """Test normalize_image with mode='none' returns float64 copy."""
    image = np.array([[100, 200]], dtype=np.uint8)
    result = normalize_image(image, mode='none')
    assert result.dtype == np.float64
    assert np.array_equal(result, image.astype(np.float64))


def test_normalize_unknown_mode_raises() -> None:
    """Test normalize_image with unknown mode raises ValueError."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="mode"):
        normalize_image(image, mode='invalid_mode')
