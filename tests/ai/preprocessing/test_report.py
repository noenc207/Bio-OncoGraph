"""Tests for preprocessing result report."""

import pytest

from bio_oncograph.ai.preprocessing.report import PreprocessingResult


def test_result_fields() -> None:
    """Test that all fields of PreprocessingResult are accessible."""
    result = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(50, 50, 3),
        transforms=("resize", "normalize"),
        warnings=("quality_low",),
        metadata={"key": "value"}
    )
    assert result.original_shape == (100, 100, 3)
    assert result.final_shape == (50, 50, 3)
    assert result.transforms == ("resize", "normalize")
    assert result.warnings == ("quality_low",)
    assert result.metadata == {"key": "value"}


def test_result_immutable() -> None:
    """Test that PreprocessingResult is frozen and prevents mutation."""
    result = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(50, 50, 3),
        transforms=("resize",),
        warnings=(),
        metadata={}
    )
    with pytest.raises(AttributeError):
        result.final_shape = (25, 25, 3)  # type: ignore


def test_result_eq_by_transforms() -> None:
    """Test that equality is based on fields including transforms."""
    result1 = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(50, 50, 3),
        transforms=("resize", "normalize"),
        warnings=(),
        metadata={}
    )
    result2 = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(50, 50, 3),
        transforms=("resize", "normalize"),
        warnings=(),
        metadata={}
    )
    assert result1 == result2


def test_result_hashable() -> None:
    """Test that PreprocessingResult is hashable and returns an integer."""
    result = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(50, 50, 3),
        transforms=("resize",),
        warnings=(),
        metadata={}
    )
    assert isinstance(hash(result), int)


def test_result_warnings_tuple() -> None:
    """Test that warnings are stored as a tuple."""
    result = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(100, 100, 3),
        transforms=(),
        warnings=("warn1", "warn2"),
        metadata={}
    )
    assert isinstance(result.warnings, tuple)
    assert result.warnings == ("warn1", "warn2")


def test_result_metadata() -> None:
    """Test that metadata dictionary is stored correctly."""
    result = PreprocessingResult(
        original_shape=(100, 100, 3),
        final_shape=(100, 100, 3),
        transforms=(),
        warnings=(),
        metadata={"source": "camera", "id": 123}
    )
    assert isinstance(result.metadata, dict)
    assert result.metadata == {"source": "camera", "id": 123}
