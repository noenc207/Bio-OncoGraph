"""Region of Interest (ROI) definitions and geometries."""

import abc
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class ROI(abc.ABC):
    """Abstract base class for Regions of Interest."""

    @abc.abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """Check if the given point is inside the ROI.

        Args:
            x: The x coordinate of the point.
            y: The y coordinate of the point.

        Returns:
            True if the point is inside, False otherwise.
        """
        pass

    @abc.abstractmethod
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get the bounding box of the ROI.

        Returns:
            A tuple of (min_x, min_y, max_x, max_y).
        """
        pass


class RectangularROI(ROI):
    """A rectangular region of interest."""

    def __init__(self, x: int, y: int, width: int, height: int):
        """Initialize a RectangularROI.

        Args:
            x: The x coordinate of the top-left corner.
            y: The y coordinate of the top-left corner.
            width: The width of the rectangle.
            height: The height of the rectangle.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within rectangle."""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get the bounding box."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class PolygonROI(ROI):
    """A polygonal region of interest."""

    def __init__(self, vertices: List[Tuple[int, int]]):
        """Initialize a PolygonROI.

        Args:
            vertices: A list of (x, y) coordinates defining the polygon.
            
        Raises:
            ValueError: If there are fewer than 3 vertices.
        """
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        self.vertices = vertices
        
        # Precompute bounding box
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        self._bbox = (min(xs), min(ys), max(xs), max(ys))

    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside polygon using ray-casting algorithm."""
        # Fast fail using bounding box
        min_x, min_y, max_x, max_y = self._bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
            
        inside = False
        n = len(self.vertices)
        p1x, p1y = self.vertices[0]
        
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersect:
                        inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get the bounding box."""
        return self._bbox


class WholeSlideROI(ROI):
    """A region of interest representing the entire slide."""

    def __init__(self, slide_width: int, slide_height: int):
        """Initialize a WholeSlideROI.
        
        Args:
            slide_width: Width of the slide at level 0.
            slide_height: Height of the slide at level 0.
        """
        self.width = slide_width
        self.height = slide_height

    def contains(self, x: float, y: float) -> bool:
        """Check if point is within the slide dimensions."""
        return 0 <= x <= self.width and 0 <= y <= self.height

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get the bounding box of the whole slide."""
        return (0, 0, self.width, self.height)
