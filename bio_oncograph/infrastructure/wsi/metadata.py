"""Data structures for WSI metadata and patch information."""

import dataclasses
from typing import Dict, Any, Tuple, Optional


@dataclasses.dataclass(frozen=True)
class SlideMetadata:
    """Metadata for a Whole Slide Image.
    
    Attributes:
        slide_id: Unique identifier for the slide.
        path: Path to the WSI file.
        format: The backend/format of the slide (e.g., 'OpenSlide', 'cuCIM').
        dimensions: (width, height) at level 0.
        level_count: Number of levels in the pyramid.
        properties: Dictionary of raw properties from the backend.
        mpp: Microns per pixel (if available).
        magnification: Objective magnification power (if available).
    """
    slide_id: str
    path: str
    format: str
    dimensions: Tuple[int, int]
    level_count: int
    properties: Dict[str, Any]
    mpp: Optional[float] = None
    magnification: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class PatchMetadata:
    """Metadata for an extracted patch.
    
    Attributes:
        slide_id: The ID of the slide this patch came from.
        coordinates: (x, y) coordinates at level 0.
        level: The pyramid level the patch was extracted from.
        patch_size: (width, height) of the patch.
        extraction_method: Descriptive string of how it was extracted (e.g., 'grid', 'random').
        timestamp: When the patch was extracted (ISO format string).
        source_metadata: Information about the source slide or process.
    """
    slide_id: str
    coordinates: Tuple[int, int]
    level: int
    patch_size: Tuple[int, int]
    extraction_method: str
    timestamp: str
    source_metadata: Dict[str, Any]
