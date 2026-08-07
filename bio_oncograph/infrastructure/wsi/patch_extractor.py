"""Patch extraction module for Whole Slide Images."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatchMetadata:
    """Metadata for an extracted patch."""
    
    slide_id: str
    x: int
    y: int
    level: int
    patch_size: Tuple[int, int]


@dataclass(frozen=True)
class PatchResult:
    """Result of a patch extraction."""
    
    patch_array: np.ndarray
    metadata: PatchMetadata


class PatchExtractor:
    """Extractor for Whole Slide Images patches."""

    def __init__(self, slide: Any) -> None:
        """Initialize with a WholeSlideImage.
        
        Args:
            slide: The whole slide image object.
        """
        self._slide = slide
        self._stats: Dict[str, Any] = {
            "num_patches_extracted": 0,
            "extraction_calls": 0,
        }

    def extract(self, x: int, y: int, level: int, patch_size: Tuple[int, int]) -> PatchResult:
        """Extract a single patch.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            level: Magnification level.
            patch_size: Size of the patch (width, height).
            
        Returns:
            PatchResult containing the image array and metadata.
        """
        logger.debug("Extracting patch at (%d, %d), level %d", x, y, level)
        try:
            # Assuming slide.read_region returns something compatible with np.array
            region = self._slide.read_region((x, y), level, patch_size)
            patch_array = np.array(region)
        except AttributeError:
            logger.warning("Slide missing read_region, returning empty array.")
            patch_array = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)

        metadata = PatchMetadata(
            slide_id=getattr(self._slide, "slide_id", "unknown"),
            x=x,
            y=y,
            level=level,
            patch_size=patch_size,
        )
        
        self._stats["num_patches_extracted"] += 1
        self._stats["extraction_calls"] += 1
        return PatchResult(patch_array=patch_array, metadata=metadata)

    def extract_grid(
        self, 
        level: int, 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int], 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[PatchResult]:
        """Extract patches in a grid pattern.
        
        Args:
            level: Magnification level.
            patch_size: Size of the patch.
            stride: Stride for extraction.
            roi: Optional Region of Interest (x, y, width, height).
            
        Returns:
            List of PatchResult objects.
        """
        self._stats["extraction_calls"] += 1
        results = []
        
        if roi is None:
            dimensions = getattr(self._slide, "dimensions", [(0, 0)])
            level_dim = dimensions[level] if level < len(dimensions) else (0, 0)
            roi = (0, 0, level_dim[0], level_dim[1])
            
        start_x, start_y, width, height = roi
        for y in range(start_y, start_y + height - patch_size[1] + 1, stride[1]):
            for x in range(start_x, start_x + width - patch_size[0] + 1, stride[0]):
                results.append(self.extract(x, y, level, patch_size))
                
        return results

    def extract_roi(
        self, 
        roi: Tuple[int, int, int, int], 
        level: int, 
        patch_size: Tuple[int, int], 
        stride: Tuple[int, int]
    ) -> List[PatchResult]:
        """Extract patches within an ROI.
        
        Args:
            roi: Region of Interest (x, y, width, height).
            level: Magnification level.
            patch_size: Size of the patch.
            stride: Stride for extraction.
            
        Returns:
            List of PatchResult objects.
        """
        return self.extract_grid(level, patch_size, stride, roi)

    def extract_random(
        self, 
        level: int, 
        patch_size: Tuple[int, int], 
        num_patches: int, 
        seed: int
    ) -> List[PatchResult]:
        """Deterministic random extraction of patches.
        
        Args:
            level: Magnification level.
            patch_size: Size of the patch.
            num_patches: Number of patches to extract.
            seed: Random seed for determinism.
            
        Returns:
            List of PatchResult objects.
        """
        self._stats["extraction_calls"] += 1
        rng = np.random.default_rng(seed)
        
        dimensions = getattr(self._slide, "dimensions", [(1000, 1000)])
        level_dim = dimensions[level] if level < len(dimensions) else (1000, 1000)
        
        max_x = max(0, level_dim[0] - patch_size[0])
        max_y = max(0, level_dim[1] - patch_size[1])
        
        results = []
        for _ in range(num_patches):
            x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
            y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
            results.append(self.extract(x, y, level, patch_size))
            
        return results

    def statistics(self) -> Dict[str, Any]:
        """Return extraction statistics.
        
        Returns:
            Dictionary of statistics.
        """
        return dict(self._stats)
