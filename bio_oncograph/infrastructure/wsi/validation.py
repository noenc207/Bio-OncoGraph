"""Validation module for Whole Slide Images."""

import logging
import os
from typing import Any, Tuple

logger = logging.getLogger(__name__)


class SlideValidator:
    """Validator for WSI properties and coordinates."""
    
    SUPPORTED_EXTENSIONS = {
        ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vms", ".scn"
    }

    @staticmethod
    def validate_coordinates(x: int, y: int, level: int, slide: Any) -> bool:
        """Check if coordinates are within bounds.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            level: Magnification level.
            slide: WholeSlideImage object.
            
        Returns:
            True if valid, False otherwise.
        """
        if not hasattr(slide, "dimensions"):
            logger.error("Slide missing dimensions attribute.")
            return False
            
        dimensions = slide.dimensions
        if level < 0 or level >= len(dimensions):
            return False
            
        w, h = dimensions[level]
        return 0 <= x < w and 0 <= y < h

    @staticmethod
    def validate_patch_bounds(
        x: int, 
        y: int, 
        patch_size: Tuple[int, int], 
        level_dimensions: Tuple[int, int]
    ) -> bool:
        """Ensure a patch fits entirely within the level dimensions.
        
        Args:
            x: X coordinate of top-left corner.
            y: Y coordinate of top-left corner.
            patch_size: Size of patch (width, height).
            level_dimensions: Dimensions of the level (width, height).
            
        Returns:
            True if the patch fits, False otherwise.
        """
        if x < 0 or y < 0:
            return False
            
        w, h = level_dimensions
        pw, ph = patch_size
        return (x + pw) <= w and (y + ph) <= h

    @staticmethod
    def validate_level(level: int, max_levels: int) -> bool:
        """Check if a magnification level is valid.
        
        Args:
            level: The level to check.
            max_levels: The maximum number of levels.
            
        Returns:
            True if valid, False otherwise.
        """
        return 0 <= level < max_levels

    @staticmethod
    def validate_roi(
        roi: Tuple[int, int, int, int], 
        slide_dimensions: Tuple[int, int]
    ) -> bool:
        """Ensure an ROI is within the slide.
        
        Args:
            roi: Region of interest (x, y, width, height).
            slide_dimensions: Total slide dimensions (width, height).
            
        Returns:
            True if valid, False otherwise.
        """
        x, y, w, h = roi
        sw, sh = slide_dimensions
        
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return False
            
        return (x + w) <= sw and (y + h) <= sh

    @classmethod
    def validate_format(cls, path: str) -> bool:
        """Check if the file format is supported.
        
        Args:
            path: Path to the slide file.
            
        Returns:
            True if the format is supported, False otherwise.
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported format: %s", ext)
            return False
        return True
