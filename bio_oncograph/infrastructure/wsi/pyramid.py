"""Pyramid structure representation for WSI."""

import dataclasses
from typing import Tuple, List
import logging

from bio_oncograph.infrastructure.wsi.slide import WholeSlideImage

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PyramidLevel:
    """Represents a single level within an image pyramid."""
    level_index: int
    dimensions: Tuple[int, int]
    downsample_factor: float


class SlidePyramid:
    """Represents the multi-resolution pyramid structure of a WSI."""

    def __init__(self, slide: WholeSlideImage):
        """Initialize the pyramid from a WholeSlideImage.

        Args:
            slide: The open WholeSlideImage to query.
            
        Raises:
            ValueError: If the slide is not open.
        """
        if not slide._is_open:
            raise ValueError("Slide must be open to initialize pyramid.")
            
        self._levels: List[PyramidLevel] = []
        
        count = slide.backend.level_count
        dims = slide.backend.level_dimensions
        downsamples = slide.backend.level_downsamples
        
        for i in range(count):
            level = PyramidLevel(
                level_index=i,
                dimensions=dims[i],
                downsample_factor=downsamples[i]
            )
            self._levels.append(level)
            
        logger.debug(f"Initialized pyramid with {count} levels.")

    @property
    def levels(self) -> List[PyramidLevel]:
        """Get the list of all pyramid levels."""
        return list(self._levels)

    def get_level(self, index: int) -> PyramidLevel:
        """Get a specific pyramid level.

        Args:
            index: The level index.

        Returns:
            The PyramidLevel at the requested index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= len(self._levels):
            raise IndexError(f"Level index {index} out of bounds.")
        return self._levels[index]
        
    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Find the best level for a given downsample factor.
        
        Args:
            downsample: The target downsample factor.
            
        Returns:
            The index of the best level to use.
        """
        best_level = 0
        for i, level in enumerate(self._levels):
            if level.downsample_factor <= downsample:
                best_level = i
            else:
                break
        return best_level
