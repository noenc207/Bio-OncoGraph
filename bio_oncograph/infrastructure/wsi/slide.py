"""WholeSlideImage wrapper for WSI backends."""

import logging
from typing import Tuple, Dict, Any, Type, Optional, List
import numpy as np

from bio_oncograph.infrastructure.wsi.backend import WSIBackend
from bio_oncograph.infrastructure.wsi.metadata import SlideMetadata

logger = logging.getLogger(__name__)


class WholeSlideImage:
    """Wrapper class for whole slide images, providing a unified interface."""

    def __init__(self, path: str, backend: WSIBackend):
        """Initialize the WholeSlideImage.

        Args:
            path: Path to the WSI file.
            backend: An instantiated WSIBackend to use for reading.
        """
        self.path = path
        self.backend = backend
        self._is_open = False

    def open(self) -> None:
        """Open the slide image using the backend."""
        logger.debug(f"Opening WSI: {self.path}")
        self.backend.open(self.path)
        self._is_open = True

    def close(self) -> None:
        """Close the slide image and release resources."""
        if self._is_open:
            logger.debug(f"Closing WSI: {self.path}")
            self.backend.close()
            self._is_open = False

    def __enter__(self) -> 'WholeSlideImage':
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Read a region from the slide at the specified level.

        Args:
            location: (x, y) coordinates at level 0.
            level: The pyramid level to read from.
            size: (width, height) of the region to read.

        Returns:
            A numpy array containing the image data.
            
        Raises:
            RuntimeError: If the slide is not open.
        """
        if not self._is_open:
            raise RuntimeError("Slide must be opened before reading.")
        return self.backend.read_region(location, level, size)

    def thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        """Get a thumbnail of the slide.

        Args:
            size: Maximum (width, height) for the thumbnail.

        Returns:
            A numpy array containing the thumbnail image.
            
        Raises:
            RuntimeError: If the slide is not open.
        """
        if not self._is_open:
            raise RuntimeError("Slide must be opened before reading.")
        return self.backend.get_thumbnail(size)

    def levels(self) -> int:
        """Get the number of levels in the image pyramid.

        Returns:
            The number of levels.
            
        Raises:
            RuntimeError: If the slide is not open.
        """
        if not self._is_open:
            raise RuntimeError("Slide must be opened before reading.")
        return self.backend.level_count
        
    def metadata(self, slide_id: str = "unknown") -> SlideMetadata:
        """Retrieve the slide metadata.
        
        Args:
            slide_id: Identifier for the slide.
            
        Returns:
            SlideMetadata object populated from backend properties.
            
        Raises:
            RuntimeError: If the slide is not open.
        """
        if not self._is_open:
            raise RuntimeError("Slide must be opened before reading.")
            
        props = self.backend.properties
        mpp = props.get("mpp-x") or props.get("mpp")
        if mpp is not None:
            try:
                mpp = float(mpp)
            except ValueError:
                mpp = None
                
        mag = props.get("objective-power")
        if mag is not None:
            try:
                mag = float(mag)
            except ValueError:
                mag = None
                
        return SlideMetadata(
            slide_id=slide_id,
            path=self.path,
            format=type(self.backend).__name__,
            dimensions=self.backend.level_dimensions[0],
            level_count=self.backend.level_count,
            properties=dict(props),
            mpp=mpp,
            magnification=mag
        )
