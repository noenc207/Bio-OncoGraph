"""
Unit tests for ImageQualityResult.

Covers: immutability, all fields, warnings tuple, metadata,
hash, repr.
"""

import pytest

from bio_oncograph.ai.image_quality.report import ImageQualityResult


def _make_result(**overrides: object) -> ImageQualityResult:
    """Helper to build a result with sensible defaults."""
    defaults = dict(
        overall_score=0.8,
        passed=True,
        blur_score=0.9,
        focus_score=0.85,
        brightness_score=1.0,
        contrast_score=0.7,
        noise_score=0.95,
        exposure_score=0.6,
        resolution_score=1.0,
        artifact_score=0.99,
        warnings=(),
        metadata={"shape": (256, 256, 3)},
    )
    defaults.update(overrides)
    return ImageQualityResult(**defaults)  # type: ignore[arg-type]


def test_result_fields() -> None:
    """All score fields are accessible."""
    r = _make_result()
    assert r.overall_score == 0.8
    assert r.passed is True
    assert r.blur_score == 0.9
    assert r.focus_score == 0.85
    assert r.brightness_score == 1.0
    assert r.contrast_score == 0.7
    assert r.noise_score == 0.95
    assert r.exposure_score == 0.6
    assert r.resolution_score == 1.0
    assert r.artifact_score == 0.99


def test_result_immutable() -> None:
    """Frozen dataclass prevents mutation."""
    r = _make_result()
    with pytest.raises(AttributeError):
        r.overall_score = 0.0  # type: ignore[misc]


def test_result_warnings() -> None:
    """Warnings are a tuple."""
    r = _make_result(warnings=("too dark",))
    assert r.warnings == ("too dark",)


def test_result_metadata() -> None:
    """Metadata dict is stored."""
    r = _make_result(metadata={"dtype": "uint8"})
    assert r.metadata["dtype"] == "uint8"


def test_result_hashable() -> None:
    """Result is hashable."""
    r = _make_result()
    assert isinstance(hash(r), int)


def test_result_failed() -> None:
    """passed=False works."""
    r = _make_result(passed=False, overall_score=0.3)
    assert r.passed is False
    assert r.overall_score == 0.3
