"""Tiling strategies for Whole Slide Images."""

import abc
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TilingStrategy(abc.ABC):
    """Abstract base class for tiling strategies."""
    
    @abc.abstractmethod
    def compute_tiles(
        self, 
        slide_dimensions: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int], 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int]]:
        """Compute tile coordinates.
        
        Args:
            slide_dimensions: Dimensions of the slide at the target level.
            patch_size: Size of the patch (width, height).
            stride: Stride between patches (x_stride, y_stride).
            roi: Optional region of interest (x, y, width, height).
            
        Returns:
            List of (x, y) coordinates for the top-left corners of tiles.
        """
        pass


class GridTiling(TilingStrategy):
    """Grid-based non-overlapping tiling strategy."""
    
    def compute_tiles(
        self, 
        slide_dimensions: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int], 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int]]:
        """Compute grid tile coordinates."""
        if roi is None:
            roi = (0, 0, slide_dimensions[0], slide_dimensions[1])
            
        start_x, start_y, width, height = roi
        
        tiles = []
        for y in range(start_y, start_y + height - patch_size[1] + 1, stride[1]):
            for x in range(start_x, start_x + width - patch_size[0] + 1, stride[0]):
                tiles.append((x, y))
                
        return tiles


class OverlappingTiling(TilingStrategy):
    """Overlapping tiling strategy."""
    
    def compute_tiles(
        self, 
        slide_dimensions: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int], 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int]]:
        """Compute overlapping tile coordinates."""
        if stride[0] >= patch_size[0] or stride[1] >= patch_size[1]:
            logger.warning("OverlappingTiling used but stride is >= patch_size")
            
        if roi is None:
            roi = (0, 0, slide_dimensions[0], slide_dimensions[1])
            
        start_x, start_y, width, height = roi
        
        tiles = []
        for y in range(start_y, start_y + height - patch_size[1] + 1, stride[1]):
            for x in range(start_x, start_x + width - patch_size[0] + 1, stride[0]):
                tiles.append((x, y))
                
        return tiles


class RandomTiling(TilingStrategy):
    """Deterministic random tiling strategy."""
    
    def __init__(self, seed: int, num_tiles: int) -> None:
        """Initialize with seed and number of tiles.
        
        Args:
            seed: Random seed.
            num_tiles: Number of tiles to generate.
        """
        self._seed = seed
        self._num_tiles = num_tiles
        
    def compute_tiles(
        self, 
        slide_dimensions: Tuple[int, int], 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int], 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int]]:
        """Compute random tile coordinates."""
        rng = np.random.default_rng(self._seed)
        
        if roi is None:
            roi = (0, 0, slide_dimensions[0], slide_dimensions[1])
            
        start_x, start_y, width, height = roi
        
        max_x = max(0, width - patch_size[0])
        max_y = max(0, height - patch_size[1])
        
        tiles = []
        for _ in range(self._num_tiles):
            x = start_x + (int(rng.integers(0, max_x + 1)) if max_x > 0 else 0)
            y = start_y + (int(rng.integers(0, max_y + 1)) if max_y > 0 else 0)
            tiles.append((x, y))
            
        return tiles
