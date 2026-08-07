"""Tests for input validation."""

import numpy as np
import pytest

from bio_oncograph.ai.preprocessing.validation import validate_preprocessing_input


def test_valid_rgb_uint8() -> None:
    """Test that a valid RGB uint8 image passes validation."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    validate_preprocessing_input(image)


def test_valid_grayscale() -> None:
    """Test that a valid grayscale image passes validation."""
    image = np.zeros((100, 100), dtype=np.uint8)
    validate_preprocessing_input(image)


def test_valid_float32() -> None:
    """Test that a valid float32 image passes validation."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    validate_preprocessing_input(image)


def test_invalid_type() -> None:
    """Test that an invalid type raises a TypeError."""
    with pytest.raises(TypeError):
        validate_preprocessing_input([1, 2, 3])  # type: ignore


def test_empty_raises() -> None:
    """Test that an empty array raises a ValueError."""
    image = np.array([])
    with pytest.raises(ValueError, match="empty"):
        validate_preprocessing_input(image)


def test_unsupported_dtype() -> None:
    """Test that an unsupported dtype raises a ValueError."""
    image = np.zeros((100, 100, 3), dtype=np.int16)
    with pytest.raises(ValueError, match="dtype"):
        validate_preprocessing_input(image)


def test_invalid_ndim() -> None:
    """Test that an array with invalid dimensions raises a ValueError."""
    image = np.zeros((100,), dtype=np.uint8)
    with pytest.raises(ValueError, match="ndim"):
        validate_preprocessing_input(image)


def test_invalid_channels() -> None:
    """Test that an array with invalid channels raises a ValueError."""
    image = np.zeros((100, 100, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="channels"):
        validate_preprocessing_input(image)


def test_nan_raises() -> None:
    """Test that an array with NaNs raises a ValueError."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_preprocessing_input(image)


def test_inf_raises() -> None:
    """Test that an array with Infs raises a ValueError."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    image[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        validate_preprocessing_input(image)


def test_below_min_resolution() -> None:
    """Test that an image below the minimum resolution raises a ValueError."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="resolution"):
        validate_preprocessing_input(image, min_resolution=(16, 16))


def test_above_max_resolution() -> None:
    """Test that an image above the maximum resolution raises a ValueError."""
    image = np.zeros((5000, 5000, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="resolution"):
        validate_preprocessing_input(image, max_resolution=(4096, 4096))
