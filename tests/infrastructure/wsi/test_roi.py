"""
Tests for WSI Regions of Interest (ROI).
"""

import pytest
from abc import ABC, abstractmethod
from typing import Tuple, List

class ROI(ABC):
    @abstractmethod
    def contains(self, point: Tuple[int, int]) -> bool:
        pass
        
    @abstractmethod
    def bounding_box(self) -> Tuple[int, int, int, int]:
        pass

class RectangularROI(ROI):
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)

    def bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

class WholeSlideROI(ROI):
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def contains(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return 0 <= px <= self.width and 0 <= py <= self.height

    def bounding_box(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.width, self.height)

class PolygonROI(ROI):
    def __init__(self, vertices: List[Tuple[int, int]]) -> None:
        self.vertices = vertices

    def contains(self, point: Tuple[int, int]) -> bool:
        # Ray casting algorithm
        x, y = point
        n = len(self.vertices)
        inside = False
        p1x, p1y = self.vertices[0]
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def bounding_box(self) -> Tuple[int, int, int, int]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)


def test_rectangular_roi_contains_inside() -> None:
    """Test RectangularROI contains internal point."""
    roi = RectangularROI(10, 10, 50, 50)
    assert roi.contains((20, 20)) is True

def test_rectangular_roi_contains_outside() -> None:
    """Test RectangularROI contains external point."""
    roi = RectangularROI(10, 10, 50, 50)
    assert roi.contains((0, 0)) is False

def test_rectangular_roi_contains_edge() -> None:
    """Test RectangularROI contains edge point."""
    roi = RectangularROI(10, 10, 50, 50)
    assert roi.contains((10, 20)) is True

def test_rectangular_roi_bounding_box() -> None:
    """Test RectangularROI bounding box."""
    roi = RectangularROI(10, 20, 30, 40)
    assert roi.bounding_box() == (10, 20, 30, 40)

def test_wholeslide_roi_contains_inside() -> None:
    """Test WholeSlideROI contains internal point."""
    roi = WholeSlideROI(1000, 1000)
    assert roi.contains((500, 500)) is True

def test_wholeslide_roi_contains_outside() -> None:
    """Test WholeSlideROI contains external point."""
    roi = WholeSlideROI(1000, 1000)
    assert roi.contains((1500, 500)) is False

def test_wholeslide_roi_bounding_box() -> None:
    """Test WholeSlideROI bounding box."""
    roi = WholeSlideROI(1000, 1000)
    assert roi.bounding_box() == (0, 0, 1000, 1000)

def test_polygon_roi_contains_inside() -> None:
    """Test PolygonROI contains internal point."""
    roi = PolygonROI([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert roi.contains((5, 5)) is True

def test_polygon_roi_contains_outside() -> None:
    """Test PolygonROI contains external point."""
    roi = PolygonROI([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert roi.contains((15, 15)) is False

def test_polygon_roi_contains_edge() -> None:
    """Test PolygonROI contains edge point."""
    roi = PolygonROI([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert roi.contains((0, 5)) is True

def test_polygon_roi_bounding_box() -> None:
    """Test PolygonROI bounding box."""
    roi = PolygonROI([(10, 10), (10, 20), (30, 20), (30, 10)])
    assert roi.bounding_box() == (10, 10, 20, 10)

def test_polygon_roi_complex_shape_inside() -> None:
    """Test complex PolygonROI contains internal point."""
    roi = PolygonROI([(0, 0), (5, 5), (10, 0), (10, 10), (0, 10)])
    assert roi.contains((5, 8)) is True

def test_polygon_roi_complex_shape_outside() -> None:
    """Test complex PolygonROI contains external point."""
    roi = PolygonROI([(0, 0), (5, 5), (10, 0), (10, 10), (0, 10)])
    assert roi.contains((5, 2)) is False

def test_rectangular_roi_zero_size() -> None:
    """Test RectangularROI with zero width/height."""
    roi = RectangularROI(10, 10, 0, 0)
    assert roi.contains((10, 10)) is True
    assert roi.contains((11, 10)) is False

def test_polygon_roi_triangle() -> None:
    """Test PolygonROI triangle shape."""
    roi = PolygonROI([(0, 0), (10, 0), (0, 10)])
    assert roi.contains((2, 2)) is True
    assert roi.contains((8, 8)) is False

def test_roi_abstract_instantiation() -> None:
    """Test ROI ABC cannot be instantiated."""
    with pytest.raises(TypeError):
        ROI()
