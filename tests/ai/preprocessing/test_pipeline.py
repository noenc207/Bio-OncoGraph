"""
Unit tests for ImagePreprocessingPipeline.

Covers: build/preprocess lifecycle, all config options, validation
errors, determinism, config override, result structure, warnings,
metadata, grayscale/float inputs.
"""

import numpy as np
import pytest

from bio_oncograph.ai.preprocessing.config import PreprocessingConfig
from bio_oncograph.ai.preprocessing.pipeline import (
    ImagePreprocessingPipeline,
)
from bio_oncograph.ai.preprocessing.report import PreprocessingResult

_RNG = np.random.RandomState(42)


def _good_image(
    h: int = 64, w: int = 64
) -> np.ndarray:
    return _RNG.randint(20, 230, (h, w, 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════
# Build & lifecycle
# ═══════════════════════════════════════════════════════════════════


def test_build_returns_self() -> None:
    """build() returns the pipeline for chaining."""
    pipe = ImagePreprocessingPipeline()
    assert pipe.build() is pipe


def test_is_built_false_initially() -> None:
    """is_built is False before build()."""
    assert ImagePreprocessingPipeline().is_built is False


def test_preprocess_without_build_raises() -> None:
    """preprocess before build raises RuntimeError."""
    with pytest.raises(RuntimeError, match="build"):
        ImagePreprocessingPipeline().preprocess(_good_image())


# ═══════════════════════════════════════════════════════════════════
# Preprocess — basic
# ═══════════════════════════════════════════════════════════════════


def test_preprocess_returns_result() -> None:
    """preprocess returns PreprocessingResult."""
    pipe = ImagePreprocessingPipeline().build()
    r = pipe.preprocess(_good_image())
    assert isinstance(r, PreprocessingResult)


def test_preprocess_output_size() -> None:
    """Output matches target_size."""
    cfg = PreprocessingConfig(target_size=(128, 128))
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert r.output_size == (128, 128)


def test_preprocess_original_size() -> None:
    """Original size is recorded."""
    pipe = ImagePreprocessingPipeline().build()
    r = pipe.preprocess(_good_image(64, 48))
    assert r.original_size == (64, 48)


def test_preprocess_applied_transforms_not_empty() -> None:
    """At least resize and normalize are applied."""
    pipe = ImagePreprocessingPipeline().build()
    r = pipe.preprocess(_good_image())
    assert len(r.applied_transforms) >= 2


# ═══════════════════════════════════════════════════════════════════
# Config overrides
# ═══════════════════════════════════════════════════════════════════


def test_config_denoise_gaussian() -> None:
    """Gaussian denoising step is added."""
    cfg = PreprocessingConfig(denoise_enabled=True, denoise_method="gaussian")
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "denoise_gaussian" in r.applied_transforms


def test_config_denoise_median() -> None:
    """Median denoising step is added."""
    cfg = PreprocessingConfig(denoise_enabled=True, denoise_method="median")
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "denoise_median" in r.applied_transforms


def test_config_denoise_bilateral() -> None:
    """Bilateral denoising step is added."""
    cfg = PreprocessingConfig(
        denoise_enabled=True, denoise_method="bilateral",
        target_size=(32, 32),
    )
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image(32, 32))
    assert "denoise_bilateral" in r.applied_transforms


def test_config_histogram_eq() -> None:
    """Histogram equalisation step is added."""
    cfg = PreprocessingConfig(histogram_equalization=True)
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "histogram_equalization" in r.applied_transforms


def test_config_clahe() -> None:
    """CLAHE step is added (overrides hist eq)."""
    cfg = PreprocessingConfig(
        clahe_enabled=True, histogram_equalization=True
    )
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "clahe" in r.applied_transforms
    assert "histogram_equalization" not in r.applied_transforms


def test_config_gamma() -> None:
    """Gamma correction step is added."""
    cfg = PreprocessingConfig(gamma=0.5)
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "gamma_correction" in r.applied_transforms


def test_config_sharpen() -> None:
    """Sharpening step is added."""
    cfg = PreprocessingConfig(sharpen_enabled=True)
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "unsharp_mask" in r.applied_transforms


def test_config_normalization_meanstd() -> None:
    """Mean/std normalisation step."""
    cfg = PreprocessingConfig(normalization_mode="meanstd")
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "normalize_meanstd" in r.applied_transforms


def test_config_normalization_none() -> None:
    """No normalisation step."""
    cfg = PreprocessingConfig(normalization_mode="none")
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "normalize_none" in r.applied_transforms


def test_config_center_crop() -> None:
    """Center crop step is added."""
    cfg = PreprocessingConfig(
        target_size=(128, 128), center_crop=(64, 64)
    )
    pipe = ImagePreprocessingPipeline(cfg).build()
    r = pipe.preprocess(_good_image())
    assert "center_crop" in r.applied_transforms


# ═══════════════════════════════════════════════════════════════════
# Validation errors
# ═══════════════════════════════════════════════════════════════════


def test_empty_image_raises() -> None:
    """Empty array raises ValueError."""
    pipe = ImagePreprocessingPipeline().build()
    with pytest.raises(ValueError):
        pipe.preprocess(np.array([], dtype=np.uint8))


def test_invalid_dtype_raises() -> None:
    """Unsupported dtype raises ValueError."""
    pipe = ImagePreprocessingPipeline().build()
    with pytest.raises(ValueError, match="dtype"):
        pipe.preprocess(np.zeros((64, 64, 3), dtype=np.int16))


def test_nan_raises() -> None:
    """NaN image raises ValueError."""
    pipe = ImagePreprocessingPipeline().build()
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pipe.preprocess(img)


def test_inf_raises() -> None:
    """Inf image raises ValueError."""
    pipe = ImagePreprocessingPipeline().build()
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        pipe.preprocess(img)


def test_too_small_raises() -> None:
    """Below min resolution raises ValueError."""
    cfg = PreprocessingConfig(min_resolution=(32, 32))
    pipe = ImagePreprocessingPipeline(cfg).build()
    with pytest.raises(ValueError):
        pipe.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))


def test_non_array_raises() -> None:
    """Non-ndarray raises TypeError."""
    pipe = ImagePreprocessingPipeline().build()
    with pytest.raises(TypeError):
        pipe.preprocess("not_image")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# validate / report
# ═══════════════════════════════════════════════════════════════════


def test_validate_valid() -> None:
    """Valid image returns True."""
    pipe = ImagePreprocessingPipeline()
    assert pipe.validate(_good_image()) is True


def test_validate_invalid() -> None:
    """Invalid image returns False."""
    pipe = ImagePreprocessingPipeline()
    assert pipe.validate(np.array([], dtype=np.uint8)) is False


def test_report_equals_preprocess() -> None:
    """report is an alias for preprocess."""
    pipe = ImagePreprocessingPipeline().build()
    img = _good_image()
    r1 = pipe.preprocess(img)
    r2 = pipe.report(img)
    assert r1 == r2


# ═══════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════


def test_deterministic_output() -> None:
    """Same image → identical output."""
    pipe = ImagePreprocessingPipeline().build()
    img = _good_image()
    r1 = pipe.preprocess(img)
    r2 = pipe.preprocess(img)
    np.testing.assert_array_equal(
        r1.processed_image, r2.processed_image
    )


# ═══════════════════════════════════════════════════════════════════
# Input types
# ═══════════════════════════════════════════════════════════════════


def test_grayscale_input() -> None:
    """2-D grayscale image is accepted."""
    pipe = ImagePreprocessingPipeline().build()
    gray = np.full((64, 64), 128, dtype=np.uint8)
    r = pipe.preprocess(gray)
    assert isinstance(r, PreprocessingResult)


def test_float_input() -> None:
    """float32 image in [0,1] is accepted."""
    pipe = ImagePreprocessingPipeline().build()
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    r = pipe.preprocess(img)
    assert isinstance(r, PreprocessingResult)


# ═══════════════════════════════════════════════════════════════════
# Metadata & config
# ═══════════════════════════════════════════════════════════════════


def test_metadata_has_original_shape() -> None:
    """Metadata includes original shape."""
    pipe = ImagePreprocessingPipeline().build()
    r = pipe.preprocess(_good_image())
    assert "original_shape" in r.metadata


def test_config_property() -> None:
    """config property returns the config."""
    cfg = PreprocessingConfig(gamma=2.0)
    pipe = ImagePreprocessingPipeline(cfg)
    assert pipe.config.gamma == 2.0


def test_repr() -> None:
    """repr includes built status."""
    pipe = ImagePreprocessingPipeline()
    assert "False" in repr(pipe)
    pipe.build()
    assert "True" in repr(pipe)
