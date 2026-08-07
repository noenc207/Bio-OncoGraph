"""
Unit tests for image quality metrics.

Covers: blur, focus, brightness, contrast, noise, exposure,
resolution, artifact — including edge cases, determinism,
and score normalisation.
"""

import numpy as np
import pytest

from bio_oncograph.ai.image_quality.metrics import (
    compute_artifact_score,
    compute_blur_score,
    compute_brightness_score,
    compute_contrast_score,
    compute_exposure_score,
    compute_focus_score,
    compute_noise_score,
    compute_resolution_score,
    to_grayscale,
)


# ── Helpers ─────────────────────────────────────────────────────────

_RNG = np.random.RandomState(42)


def _sharp_image() -> np.ndarray:
    """256×256 RGB with strong edges (checkerboard)."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            if (r // 16 + c // 16) % 2 == 0:
                img[r, c] = [200, 200, 200]
            else:
                img[r, c] = [50, 50, 50]
    return img


def _uniform_image(value: int = 128) -> np.ndarray:
    """Uniform (perfectly flat) 128×128 image."""
    return np.full((128, 128, 3), value, dtype=np.uint8)


def _noisy_image() -> np.ndarray:
    """Image with heavy Gaussian noise."""
    base = np.full((128, 128), 128.0, dtype=np.float64)
    noise = _RNG.normal(0, 80, size=(128, 128))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ── to_grayscale ───────────────────────────────────────────────────


def test_grayscale_2d() -> None:
    """2-D array passes through."""
    g = np.zeros((64, 64), dtype=np.float64)
    assert to_grayscale(g).ndim == 2


def test_grayscale_3ch() -> None:
    """3-channel array converts to 2-D."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert to_grayscale(img).ndim == 2


def test_grayscale_1ch() -> None:
    """Single-channel (H,W,1) squeezes to 2-D."""
    img = np.zeros((64, 64, 1), dtype=np.uint8)
    assert to_grayscale(img).ndim == 2


def test_grayscale_float01() -> None:
    """Float [0,1] images are scaled to [0,255]."""
    img = np.full((64, 64), 0.5, dtype=np.float64)
    gray = to_grayscale(img)
    assert gray.max() == pytest.approx(127.5)


# ── Blur score ──────────────────────────────────────────────────────


def test_blur_sharp() -> None:
    """Sharp image gets high blur score."""
    score = compute_blur_score(_sharp_image(), threshold=100.0)
    assert score > 0.5


def test_blur_uniform() -> None:
    """Uniform image gets near-zero blur score."""
    score = compute_blur_score(_uniform_image(), threshold=100.0)
    assert score < 0.01


def test_blur_deterministic() -> None:
    """Same image → same score."""
    img = _sharp_image()
    assert compute_blur_score(img) == compute_blur_score(img)


# ── Focus score ─────────────────────────────────────────────────────


def test_focus_sharp() -> None:
    """Sharp image has high focus score."""
    score = compute_focus_score(_sharp_image())
    assert score > 0.5


def test_focus_uniform() -> None:
    """Uniform image has near-zero focus."""
    score = compute_focus_score(_uniform_image())
    assert score < 0.01


# ── Brightness score ────────────────────────────────────────────────


def test_brightness_mid_range() -> None:
    """Mid-grey image scores 1.0."""
    score = compute_brightness_score(
        _uniform_image(128), brightness_range=(0.1, 0.9)
    )
    assert score == pytest.approx(1.0)


def test_brightness_too_dark() -> None:
    """Very dark image scores low."""
    score = compute_brightness_score(
        _uniform_image(5), brightness_range=(0.1, 0.9)
    )
    assert score < 0.3


def test_brightness_too_bright() -> None:
    """Very bright image scores low."""
    score = compute_brightness_score(
        _uniform_image(250), brightness_range=(0.1, 0.9)
    )
    assert score < 0.3


# ── Contrast score ──────────────────────────────────────────────────


def test_contrast_high() -> None:
    """Checkerboard has high contrast."""
    score = compute_contrast_score(_sharp_image(), threshold=0.1)
    assert score > 0.5


def test_contrast_uniform() -> None:
    """Uniform image has zero contrast."""
    score = compute_contrast_score(_uniform_image(), threshold=0.1)
    assert score == pytest.approx(0.0, abs=1e-6)


# ── Noise score ─────────────────────────────────────────────────────


def test_noise_clean() -> None:
    """Clean (uniform) image gets high noise score."""
    score = compute_noise_score(_uniform_image(), threshold=50.0)
    assert score > 0.9


def test_noise_noisy() -> None:
    """Noisy image gets lower noise score."""
    img = _noisy_image()
    score = compute_noise_score(
        img[:, :, np.newaxis].repeat(3, axis=2),
        threshold=50.0,
    )
    assert score < 0.8


# ── Exposure score ──────────────────────────────────────────────────


def test_exposure_good() -> None:
    """Image with wide histogram has decent exposure."""
    img = _RNG.randint(20, 235, (128, 128, 3), dtype=np.uint8)
    score = compute_exposure_score(img)
    assert score > 0.3


def test_exposure_clipped() -> None:
    """Fully saturated image has poor exposure."""
    img = np.full((128, 128, 3), 255, dtype=np.uint8)
    score = compute_exposure_score(img)
    assert score < 0.1


# ── Resolution score ───────────────────────────────────────────────


def test_resolution_adequate() -> None:
    """Image meeting minimum scores 1.0."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    score = compute_resolution_score(img, (64, 64))
    assert score == pytest.approx(1.0)


def test_resolution_too_small() -> None:
    """Image below minimum scores < 1.0."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    score = compute_resolution_score(img, (64, 64))
    assert score == pytest.approx(0.5)


# ── Artifact score ──────────────────────────────────────────────────


def test_artifact_clean() -> None:
    """Mid-grey image has no artifacts."""
    score = compute_artifact_score(_uniform_image(128))
    assert score == pytest.approx(1.0)


def test_artifact_saturated() -> None:
    """Fully white image is all-saturated → low score."""
    img = np.full((64, 64, 3), 255, dtype=np.uint8)
    score = compute_artifact_score(img)
    assert score < 0.1
