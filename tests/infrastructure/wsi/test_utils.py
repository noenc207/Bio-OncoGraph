import pytest
import numpy as np
from typing import List, Tuple

def compute_grid_coordinates(dimensions: Tuple[int, int], patch_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    coords = []
    for y in range(0, dimensions[1], patch_size[1]):
        for x in range(0, dimensions[0], patch_size[0]):
            coords.append((x, y))
    return coords

def estimate_tissue_fraction(patch: np.ndarray, threshold: int = 200) -> float:
    if patch.size == 0:
        return 0.0
    gray = np.mean(patch, axis=-1)
    tissue_pixels = np.sum(gray < threshold)
    return float(tissue_pixels) / gray.size

def patch_to_rgb(patch: np.ndarray) -> np.ndarray:
    if patch.shape[-1] == 4:
        return patch[..., :3]
    return patch

# 12 tests
def test_compute_grid_basic() -> None:
    """Test compute grid basic."""
    coords = compute_grid_coordinates((100, 100), (50, 50))
    assert len(coords) == 4

def test_compute_grid_exact() -> None:
    """Test compute grid exact coords."""
    coords = compute_grid_coordinates((100, 100), (100, 100))
    assert coords == [(0, 0)]

def test_compute_grid_non_divisible() -> None:
    """Test compute grid non divisible."""
    coords = compute_grid_coordinates((100, 100), (60, 60))
    assert len(coords) == 4

def test_compute_grid_zero_dim() -> None:
    """Test compute grid zero dimension."""
    coords = compute_grid_coordinates((0, 0), (50, 50))
    assert len(coords) == 0

def test_estimate_tissue_empty() -> None:
    """Test tissue fraction empty."""
    patch = np.zeros((0, 0, 3))
    assert estimate_tissue_fraction(patch) == 0.0

def test_estimate_tissue_all_tissue() -> None:
    """Test tissue fraction all tissue."""
    patch = np.zeros((10, 10, 3), dtype=np.uint8)
    assert estimate_tissue_fraction(patch) == 1.0

def test_estimate_tissue_all_background() -> None:
    """Test tissue fraction all background."""
    patch = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert estimate_tissue_fraction(patch) == 0.0

def test_estimate_tissue_half() -> None:
    """Test tissue fraction half."""
    patch = np.full((10, 10, 3), 255, dtype=np.uint8)
    patch[:5, :, :] = 0
    assert estimate_tissue_fraction(patch) == 0.5

def test_estimate_tissue_custom_threshold() -> None:
    """Test tissue fraction custom threshold."""
    patch = np.full((10, 10, 3), 150, dtype=np.uint8)
    assert estimate_tissue_fraction(patch, threshold=100) == 0.0
    assert estimate_tissue_fraction(patch, threshold=200) == 1.0

def test_patch_to_rgb_from_rgba() -> None:
    """Test RGBA to RGB."""
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgb = patch_to_rgb(rgba)
    assert rgb.shape == (10, 10, 3)

def test_patch_to_rgb_from_rgb() -> None:
    """Test RGB to RGB."""
    rgb_in = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_out = patch_to_rgb(rgb_in)
    assert rgb_out.shape == (10, 10, 3)

def test_patch_to_rgb_noop() -> None:
    """Test patch to RGB noop identity."""
    rgb_in = np.zeros((10, 10, 3), dtype=np.uint8)
    assert patch_to_rgb(rgb_in) is rgb_in
