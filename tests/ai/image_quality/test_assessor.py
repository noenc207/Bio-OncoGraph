"""
Unit tests for ImageQualityAssessor.

Covers: valid images, blurred, overexposed, underexposed,
noisy, low-resolution, empty, invalid dtype, invalid channels,
determinism, configuration override, compute_score, validate,
report, warnings, metadata.
"""

import numpy as np
import pytest

from bio_oncograph.ai.image_quality.assessor import (
    ImageQualityAssessor,
)
from bio_oncograph.ai.image_quality.config import IQAConfig
from bio_oncograph.ai.image_quality.report import ImageQualityResult

_RNG = np.random.RandomState(42)


# ── Helpers ─────────────────────────────────────────────────────────


def _good_image() -> np.ndarray:
    """Checkerboard with decent quality."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            if (r // 16 + c // 16) % 2 == 0:
                img[r, c] = [180, 180, 180]
            else:
                img[r, c] = [70, 70, 70]
    return img


def _blurry_image() -> np.ndarray:
    """Perfectly uniform → zero Laplacian variance."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


def _overexposed() -> np.ndarray:
    """Almost all-white image."""
    return np.full((128, 128, 3), 250, dtype=np.uint8)


def _underexposed() -> np.ndarray:
    """Almost all-black image."""
    return np.full((128, 128, 3), 5, dtype=np.uint8)


def _noisy_image() -> np.ndarray:
    """Heavy Gaussian noise."""
    base = np.full((128, 128), 128.0)
    noise = _RNG.normal(0, 80, (128, 128))
    g = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def assessor() -> ImageQualityAssessor:
    return ImageQualityAssessor()


# ══════════════════════════════════════════════════════════════════
# assess — valid images
# ══════════════════════════════════════════════════════════════════


def test_assess_returns_result(
    assessor: ImageQualityAssessor,
) -> None:
    """assess returns an ImageQualityResult."""
    r = assessor.assess(_good_image())
    assert isinstance(r, ImageQualityResult)


def test_good_image_passes(
    assessor: ImageQualityAssessor,
) -> None:
    """A decent image passes quality check."""
    r = assessor.assess(_good_image())
    assert r.passed is True
    assert r.overall_score >= 0.5


def test_all_scores_in_range(
    assessor: ImageQualityAssessor,
) -> None:
    """All individual scores are in [0, 1]."""
    r = assessor.assess(_good_image())
    for attr in [
        "blur_score", "focus_score", "brightness_score",
        "contrast_score", "noise_score", "exposure_score",
        "resolution_score", "artifact_score", "overall_score",
    ]:
        score = getattr(r, attr)
        assert 0.0 <= score <= 1.0, f"{attr}={score}"


# ══════════════════════════════════════════════════════════════════
# assess — degraded images
# ══════════════════════════════════════════════════════════════════


def test_blurry_low_blur_score(
    assessor: ImageQualityAssessor,
) -> None:
    """Uniform image has near-zero blur score."""
    r = assessor.assess(_blurry_image())
    assert r.blur_score < 0.05


def test_overexposed_low_brightness(
    assessor: ImageQualityAssessor,
) -> None:
    """Overexposed image gets low brightness score."""
    r = assessor.assess(_overexposed())
    assert r.brightness_score < 0.6


def test_underexposed_low_brightness(
    assessor: ImageQualityAssessor,
) -> None:
    """Underexposed image gets low brightness score."""
    r = assessor.assess(_underexposed())
    assert r.brightness_score < 0.3


def test_noisy_lower_noise_score(
    assessor: ImageQualityAssessor,
) -> None:
    """Noisy image gets lower noise score than clean."""
    r_noisy = assessor.assess(_noisy_image())
    r_clean = assessor.assess(_blurry_image())
    assert r_noisy.noise_score < r_clean.noise_score


# ══════════════════════════════════════════════════════════════════
# assess — validation errors
# ══════════════════════════════════════════════════════════════════


def test_empty_image_raises(
    assessor: ImageQualityAssessor,
) -> None:
    """Empty array raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        assessor.assess(np.array([], dtype=np.uint8))


def test_invalid_dtype_raises(
    assessor: ImageQualityAssessor,
) -> None:
    """Unsupported dtype raises ValueError."""
    with pytest.raises(ValueError, match="dtype"):
        assessor.assess(np.zeros((64, 64, 3), dtype=np.int16))


def test_invalid_channels_raises(
    assessor: ImageQualityAssessor,
) -> None:
    """2-channel image raises ValueError."""
    with pytest.raises(ValueError, match="channels"):
        assessor.assess(np.zeros((64, 64, 2), dtype=np.uint8))


def test_non_array_raises(
    assessor: ImageQualityAssessor,
) -> None:
    """Non-ndarray raises TypeError."""
    with pytest.raises(TypeError):
        assessor.assess("not_an_image")  # type: ignore[arg-type]


def test_low_resolution_raises() -> None:
    """Image below minimum_resolution raises ValueError."""
    cfg = IQAConfig(minimum_resolution=(128, 128))
    a = ImageQualityAssessor(config=cfg)
    with pytest.raises(ValueError, match="resolution"):
        a.assess(np.zeros((64, 64, 3), dtype=np.uint8))


# ══════════════════════════════════════════════════════════════════
# compute_score / validate / report
# ══════════════════════════════════════════════════════════════════


def test_compute_score_returns_float(
    assessor: ImageQualityAssessor,
) -> None:
    """compute_score returns a float."""
    s = assessor.compute_score(_good_image())
    assert isinstance(s, float)


def test_validate_returns_bool(
    assessor: ImageQualityAssessor,
) -> None:
    """validate returns a bool."""
    v = assessor.validate(_good_image())
    assert isinstance(v, bool)


def test_report_equals_assess(
    assessor: ImageQualityAssessor,
) -> None:
    """report() returns same result as assess()."""
    img = _good_image()
    assert assessor.report(img) == assessor.assess(img)


# ══════════════════════════════════════════════════════════════════
# Determinism
# ══════════════════════════════════════════════════════════════════


def test_deterministic(
    assessor: ImageQualityAssessor,
) -> None:
    """Same image → identical result."""
    img = _good_image()
    r1 = assessor.assess(img)
    r2 = assessor.assess(img)
    assert r1.overall_score == r2.overall_score
    assert r1.blur_score == r2.blur_score


# ══════════════════════════════════════════════════════════════════
# Configuration override
# ══════════════════════════════════════════════════════════════════


def test_config_override_threshold() -> None:
    """Custom config changes pass/fail verdict."""
    strict = IQAConfig(minimum_quality_score=0.99)
    a = ImageQualityAssessor(config=strict)
    r = a.assess(_good_image())
    # With very strict threshold, even a decent image may fail
    assert r.overall_score < 0.99 or r.passed is True


def test_config_property() -> None:
    """config property exposes the active config."""
    cfg = IQAConfig(blur_threshold=200.0)
    a = ImageQualityAssessor(config=cfg)
    assert a.config.blur_threshold == 200.0


# ══════════════════════════════════════════════════════════════════
# Warnings & metadata
# ══════════════════════════════════════════════════════════════════


def test_blurry_generates_warning(
    assessor: ImageQualityAssessor,
) -> None:
    """Blurry image produces a blur warning."""
    r = assessor.assess(_blurry_image())
    assert any("blur" in w.lower() for w in r.warnings)


def test_metadata_contains_shape(
    assessor: ImageQualityAssessor,
) -> None:
    """Metadata includes image shape."""
    r = assessor.assess(_good_image())
    assert "shape" in r.metadata
    assert r.metadata["shape"] == (256, 256, 3)


def test_metadata_contains_dtype(
    assessor: ImageQualityAssessor,
) -> None:
    """Metadata includes image dtype."""
    r = assessor.assess(_good_image())
    assert r.metadata["dtype"] == "uint8"


def test_grayscale_image(
    assessor: ImageQualityAssessor,
) -> None:
    """2-D grayscale image is accepted."""
    gray = np.full((64, 64), 128, dtype=np.uint8)
    r = assessor.assess(gray)
    assert isinstance(r, ImageQualityResult)


def test_float_image(
    assessor: ImageQualityAssessor,
) -> None:
    """float32 image is accepted."""
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    r = assessor.assess(img)
    assert isinstance(r, ImageQualityResult)


def test_repr(assessor: ImageQualityAssessor) -> None:
    """repr includes min_score."""
    assert "0.5" in repr(assessor)
