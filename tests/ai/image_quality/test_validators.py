"""
Unit tests for image validators.

Covers: valid image, empty, wrong type, unsupported dtype,
invalid channels, NaN, low resolution, dimensions, and
the aggregate validate_image function.
"""

import numpy as np
import pytest

from bio_oncograph.ai.image_quality.validators import (
    validate_channels,
    validate_dimensions,
    validate_dtype,
    validate_image,
    validate_no_nan,
    validate_not_empty,
    validate_resolution,
    validate_type,
)


# ══════════════════════════════════════════════════════════════════
# validate_type
# ══════════════════════════════════════════════════════════════════


def test_type_valid() -> None:
    """ndarray passes."""
    validate_type(np.zeros((4, 4), dtype=np.uint8))


def test_type_invalid() -> None:
    """Non-ndarray raises TypeError."""
    with pytest.raises(TypeError, match="numpy.ndarray"):
        validate_type([[1, 2], [3, 4]])  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════
# validate_not_empty
# ══════════════════════════════════════════════════════════════════


def test_not_empty_valid() -> None:
    """Non-empty array passes."""
    validate_not_empty(np.zeros((2, 2), dtype=np.uint8))


def test_not_empty_invalid() -> None:
    """Empty array raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        validate_not_empty(np.array([], dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════
# validate_dtype
# ══════════════════════════════════════════════════════════════════


def test_dtype_uint8() -> None:
    """uint8 is supported."""
    validate_dtype(np.zeros(4, dtype=np.uint8))


def test_dtype_float32() -> None:
    """float32 is supported."""
    validate_dtype(np.zeros(4, dtype=np.float32))


def test_dtype_float64() -> None:
    """float64 is supported."""
    validate_dtype(np.zeros(4, dtype=np.float64))


def test_dtype_int32_raises() -> None:
    """int32 is unsupported."""
    with pytest.raises(ValueError, match="dtype"):
        validate_dtype(np.zeros(4, dtype=np.int32))


# ══════════════════════════════════════════════════════════════════
# validate_dimensions
# ══════════════════════════════════════════════════════════════════


def test_dim_2d() -> None:
    """2-D array passes."""
    validate_dimensions(np.zeros((4, 4), dtype=np.uint8))


def test_dim_3d() -> None:
    """3-D array passes."""
    validate_dimensions(np.zeros((4, 4, 3), dtype=np.uint8))


def test_dim_1d_raises() -> None:
    """1-D array raises ValueError."""
    with pytest.raises(ValueError, match="ndim"):
        validate_dimensions(np.zeros(4, dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════
# validate_channels
# ══════════════════════════════════════════════════════════════════


def test_channels_3() -> None:
    """3 channels passes."""
    validate_channels(np.zeros((4, 4, 3), dtype=np.uint8))


def test_channels_1() -> None:
    """1 channel passes."""
    validate_channels(np.zeros((4, 4, 1), dtype=np.uint8))


def test_channels_4() -> None:
    """4 channels (RGBA) passes."""
    validate_channels(np.zeros((4, 4, 4), dtype=np.uint8))


def test_channels_2_raises() -> None:
    """2 channels raises ValueError."""
    with pytest.raises(ValueError, match="channels"):
        validate_channels(np.zeros((4, 4, 2), dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════
# validate_no_nan
# ══════════════════════════════════════════════════════════════════


def test_no_nan_clean() -> None:
    """Float array without NaN passes."""
    validate_no_nan(np.zeros((4, 4), dtype=np.float32))


def test_no_nan_with_nan() -> None:
    """Float array with NaN raises ValueError."""
    img = np.zeros((4, 4), dtype=np.float32)
    img[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_no_nan(img)


def test_no_nan_integer_skipped() -> None:
    """Integer arrays skip NaN check."""
    validate_no_nan(np.zeros((4, 4), dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════
# validate_resolution
# ══════════════════════════════════════════════════════════════════


def test_resolution_adequate() -> None:
    """Image meeting minimum passes."""
    validate_resolution(
        np.zeros((64, 64), dtype=np.uint8), (32, 32)
    )


def test_resolution_too_small() -> None:
    """Image below minimum raises ValueError."""
    with pytest.raises(ValueError, match="resolution"):
        validate_resolution(
            np.zeros((16, 16), dtype=np.uint8), (32, 32)
        )


# ══════════════════════════════════════════════════════════════════
# validate_image (aggregate)
# ══════════════════════════════════════════════════════════════════


def test_validate_image_valid() -> None:
    """Valid RGB uint8 image passes all checks."""
    validate_image(
        np.zeros((64, 64, 3), dtype=np.uint8), (32, 32)
    )


def test_validate_image_catches_type() -> None:
    """Aggregate catches type error."""
    with pytest.raises(TypeError):
        validate_image("not_array")  # type: ignore[arg-type]


def test_validate_image_catches_empty() -> None:
    """Aggregate catches empty array."""
    with pytest.raises(ValueError):
        validate_image(np.array([], dtype=np.uint8))
