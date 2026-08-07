"""
Tests for SlideCoordinate and CoordinateTransformer.
"""

import pytest
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SlideCoordinate:
    x: int
    y: int
    level: int

class CoordinateTransformer:
    def __init__(self, downsamples: list[float]) -> None:
        self.downsamples = downsamples

    def level_to_level(self, coord: SlideCoordinate, target_level: int) -> SlideCoordinate:
        if target_level < 0 or target_level >= len(self.downsamples):
            raise ValueError(f"Invalid target level {target_level}")
        if coord.level < 0 or coord.level >= len(self.downsamples):
            raise ValueError(f"Invalid source level {coord.level}")
            
        source_ds = self.downsamples[coord.level]
        target_ds = self.downsamples[target_level]
        
        factor = source_ds / target_ds
        new_x = int(coord.x * factor)
        new_y = int(coord.y * factor)
        
        return SlideCoordinate(new_x, new_y, target_level)


def test_slide_coordinate_creation() -> None:
    """Test SlideCoordinate creation."""
    coord = SlideCoordinate(x=100, y=200, level=0)
    assert coord.x == 100
    assert coord.y == 200
    assert coord.level == 0

def test_transformer_initialization() -> None:
    """Test CoordinateTransformer initialization."""
    transformer = CoordinateTransformer([1.0, 2.0, 4.0])
    assert len(transformer.downsamples) == 3

def test_level_to_level_same_level() -> None:
    """Test conversion to the same level."""
    transformer = CoordinateTransformer([1.0, 2.0, 4.0])
    coord = SlideCoordinate(100, 200, 0)
    new_coord = transformer.level_to_level(coord, 0)
    assert new_coord.x == 100
    assert new_coord.y == 200
    assert new_coord.level == 0

def test_level_to_level_downsample() -> None:
    """Test conversion to a lower resolution level (higher level index)."""
    transformer = CoordinateTransformer([1.0, 2.0, 4.0])
    coord = SlideCoordinate(100, 200, 0)
    new_coord = transformer.level_to_level(coord, 1)
    assert new_coord.x == 50
    assert new_coord.y == 100
    assert new_coord.level == 1

def test_level_to_level_upsample() -> None:
    """Test conversion to a higher resolution level (lower level index)."""
    transformer = CoordinateTransformer([1.0, 2.0, 4.0])
    coord = SlideCoordinate(50, 100, 1)
    new_coord = transformer.level_to_level(coord, 0)
    assert new_coord.x == 100
    assert new_coord.y == 200
    assert new_coord.level == 0

def test_invalid_target_level_high() -> None:
    """Test exception on invalid high target level."""
    transformer = CoordinateTransformer([1.0, 2.0])
    coord = SlideCoordinate(10, 10, 0)
    with pytest.raises(ValueError):
        transformer.level_to_level(coord, 2)

def test_invalid_target_level_low() -> None:
    """Test exception on invalid low target level."""
    transformer = CoordinateTransformer([1.0, 2.0])
    coord = SlideCoordinate(10, 10, 0)
    with pytest.raises(ValueError):
        transformer.level_to_level(coord, -1)

def test_invalid_source_level() -> None:
    """Test exception on invalid source level."""
    transformer = CoordinateTransformer([1.0, 2.0])
    coord = SlideCoordinate(10, 10, 5)
    with pytest.raises(ValueError):
        transformer.level_to_level(coord, 0)

def test_large_coordinates() -> None:
    """Test with very large coordinates."""
    transformer = CoordinateTransformer([1.0, 4.0])
    coord = SlideCoordinate(100000, 200000, 0)
    new_coord = transformer.level_to_level(coord, 1)
    assert new_coord.x == 25000
    assert new_coord.y == 50000

def test_truncation_behavior() -> None:
    """Test integer truncation during conversion."""
    transformer = CoordinateTransformer([1.0, 3.0])
    coord = SlideCoordinate(10, 10, 0)
    new_coord = transformer.level_to_level(coord, 1)
    # 10 / 3 = 3.333 -> int is 3
    assert new_coord.x == 3
    assert new_coord.y == 3

def test_zero_coordinates() -> None:
    """Test origin coordinates conversion."""
    transformer = CoordinateTransformer([1.0, 2.0])
    coord = SlideCoordinate(0, 0, 0)
    new_coord = transformer.level_to_level(coord, 1)
    assert new_coord.x == 0
    assert new_coord.y == 0

def test_coordinate_equality() -> None:
    """Test coordinate dataclass equality."""
    coord1 = SlideCoordinate(10, 20, 0)
    coord2 = SlideCoordinate(10, 20, 0)
    assert coord1 == coord2

def test_coordinate_inequality() -> None:
    """Test coordinate dataclass inequality."""
    coord1 = SlideCoordinate(10, 20, 0)
    coord2 = SlideCoordinate(10, 20, 1)
    assert coord1 != coord2

def test_level_to_level_multiple_steps() -> None:
    """Test conversion jumping multiple levels."""
    transformer = CoordinateTransformer([1.0, 2.0, 4.0, 16.0])
    coord = SlideCoordinate(160, 320, 0)
    new_coord = transformer.level_to_level(coord, 3)
    assert new_coord.x == 10
    assert new_coord.y == 20

def test_float_downsamples_precision() -> None:
    """Test conversion with non-integer downsamples."""
    transformer = CoordinateTransformer([1.0, 2.5, 5.0])
    coord = SlideCoordinate(100, 100, 0)
    new_coord = transformer.level_to_level(coord, 1)
    assert new_coord.x == 40
    
def test_negative_coordinates() -> None:
    """Test behavior with negative coordinates (if applicable)."""
    transformer = CoordinateTransformer([1.0, 2.0])
    coord = SlideCoordinate(-10, -20, 0)
    new_coord = transformer.level_to_level(coord, 1)
    assert new_coord.x == -5
    assert new_coord.y == -10
