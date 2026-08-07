"""
Tests for SlidePyramid and PyramidLevel dataclass.
"""

import pytest
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PyramidLevel:
    level: int
    dimensions: Tuple[int, int]
    downsample: float

class SlidePyramid:
    def __init__(self, dimensions: List[Tuple[int, int]], downsamples: List[float]) -> None:
        if len(dimensions) != len(downsamples):
            raise ValueError("Dimensions and downsamples must have the same length")
        self.levels: List[PyramidLevel] = [
            PyramidLevel(i, dim, ds) for i, (dim, ds) in enumerate(zip(dimensions, downsamples))
        ]
        
    def get_level(self, level: int) -> PyramidLevel:
        if level < 0 or level >= len(self.levels):
            raise ValueError(f"Level {level} out of bounds")
        return self.levels[level]
        
    def get_best_level_for_downsample(self, downsample: float) -> int:
        best_level = 0
        for i, lvl in enumerate(self.levels):
            if lvl.downsample <= downsample:
                best_level = i
            else:
                break
        return best_level

    @property
    def level_count(self) -> int:
        return len(self.levels)


def test_pyramid_level_dataclass() -> None:
    """Test PyramidLevel dataclass instantiation."""
    lvl = PyramidLevel(level=0, dimensions=(1000, 1000), downsample=1.0)
    assert lvl.level == 0
    assert lvl.dimensions == (1000, 1000)
    assert lvl.downsample == 1.0

def test_slide_pyramid_initialization() -> None:
    """Test SlidePyramid initialization."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500)], [1.0, 2.0])
    assert pyramid.level_count == 2
    assert pyramid.levels[0].dimensions == (1000, 1000)

def test_slide_pyramid_mismatched_lengths() -> None:
    """Test SlidePyramid initialization with mismatched lists raises error."""
    with pytest.raises(ValueError):
        SlidePyramid([(1000, 1000)], [1.0, 2.0])

def test_get_level_valid() -> None:
    """Test getting a valid level."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500)], [1.0, 2.0])
    lvl = pyramid.get_level(1)
    assert lvl.level == 1
    assert lvl.downsample == 2.0

def test_get_level_out_of_bounds_high() -> None:
    """Test getting a level too high raises error."""
    pyramid = SlidePyramid([(1000, 1000)], [1.0])
    with pytest.raises(ValueError):
        pyramid.get_level(1)

def test_get_level_out_of_bounds_low() -> None:
    """Test getting a level too low raises error."""
    pyramid = SlidePyramid([(1000, 1000)], [1.0])
    with pytest.raises(ValueError):
        pyramid.get_level(-1)

def test_get_best_level_exact_match() -> None:
    """Test get_best_level_for_downsample with exact match."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500), (250, 250)], [1.0, 2.0, 4.0])
    assert pyramid.get_best_level_for_downsample(2.0) == 1

def test_get_best_level_between_levels() -> None:
    """Test get_best_level_for_downsample between levels."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500), (250, 250)], [1.0, 2.0, 4.0])
    assert pyramid.get_best_level_for_downsample(3.0) == 1

def test_get_best_level_high_downsample() -> None:
    """Test get_best_level_for_downsample with very high downsample."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500), (250, 250)], [1.0, 2.0, 4.0])
    assert pyramid.get_best_level_for_downsample(10.0) == 2

def test_get_best_level_low_downsample() -> None:
    """Test get_best_level_for_downsample with downsample < 1.0."""
    pyramid = SlidePyramid([(1000, 1000), (500, 500)], [1.0, 2.0])
    assert pyramid.get_best_level_for_downsample(0.5) == 0

def test_level_count_property() -> None:
    """Test level_count property."""
    pyramid = SlidePyramid([(1000, 1000)], [1.0])
    assert pyramid.level_count == 1

def test_pyramid_level_equality() -> None:
    """Test equality between identical PyramidLevel objects."""
    lvl1 = PyramidLevel(0, (100, 100), 1.0)
    lvl2 = PyramidLevel(0, (100, 100), 1.0)
    assert lvl1 == lvl2

def test_pyramid_level_inequality() -> None:
    """Test inequality between different PyramidLevel objects."""
    lvl1 = PyramidLevel(0, (100, 100), 1.0)
    lvl2 = PyramidLevel(1, (50, 50), 2.0)
    assert lvl1 != lvl2
