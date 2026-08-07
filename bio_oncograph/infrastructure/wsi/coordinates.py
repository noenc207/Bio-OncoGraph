"""Coordinate structures and transformations for WSI."""

import dataclasses
from typing import Tuple

@dataclasses.dataclass(frozen=True)
class SlideCoordinate:
    """Represents a coordinate point on a specific pyramid level.
    
    Attributes:
        x: The x coordinate (width dimension).
        y: The y coordinate (height dimension).
        level: The pyramid level this coordinate belongs to.
    """
    x: int
    y: int
    level: int


class CoordinateTransformer:
    """Transforms coordinates between pyramid levels."""
    
    def __init__(self, downsamples: Tuple[float, ...]):
        """Initialize the transformer.
        
        Args:
            downsamples: The downsample factor for each level.
        """
        self._downsamples = downsamples
        
    def to_level_0(self, coord: SlideCoordinate) -> SlideCoordinate:
        """Transform a coordinate to level 0.
        
        Args:
            coord: The input coordinate.
            
        Returns:
            The transformed coordinate at level 0.
            
        Raises:
            ValueError: If the coordinate's level is invalid.
        """
        if coord.level < 0 or coord.level >= len(self._downsamples):
            raise ValueError(f"Invalid level: {coord.level}")
            
        if coord.level == 0:
            return coord
            
        factor = self._downsamples[coord.level]
        x_0 = int(round(coord.x * factor))
        y_0 = int(round(coord.y * factor))
        return SlideCoordinate(x=x_0, y=y_0, level=0)
        
    def transform(self, coord: SlideCoordinate, target_level: int) -> SlideCoordinate:
        """Transform a coordinate to a target level.
        
        Args:
            coord: The input coordinate.
            target_level: The target level index.
            
        Returns:
            The transformed coordinate.
            
        Raises:
            ValueError: If either level is invalid.
        """
        if coord.level < 0 or coord.level >= len(self._downsamples):
            raise ValueError(f"Invalid source level: {coord.level}")
        if target_level < 0 or target_level >= len(self._downsamples):
            raise ValueError(f"Invalid target level: {target_level}")
            
        if coord.level == target_level:
            return coord
            
        # Transform to level 0 first, then to target
        source_factor = self._downsamples[coord.level]
        target_factor = self._downsamples[target_level]
        
        x_target = int(round((coord.x * source_factor) / target_factor))
        y_target = int(round((coord.y * source_factor) / target_factor))
        
        return SlideCoordinate(x=x_target, y=y_target, level=target_level)
