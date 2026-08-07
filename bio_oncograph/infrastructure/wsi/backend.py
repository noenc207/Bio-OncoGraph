"""WSI Backend abstract base class and implementations."""

import abc
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WSIBackend(abc.ABC):
    """Abstract base class for Whole Slide Image backends."""

    @abc.abstractmethod
    def open(self, path: str) -> None:
        """Open a WSI file.

        Args:
            path: Path to the WSI file.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the WSI file and release resources."""
        pass

    @abc.abstractmethod
    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Read a region from the slide at the specified level.

        Args:
            location: (x, y) coordinates at level 0.
            level: The pyramid level to read from.
            size: (width, height) of the region to read.

        Returns:
            A numpy array containing the image data (H, W, C).
        """
        pass

    @abc.abstractmethod
    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        """Get a thumbnail image of the slide.

        Args:
            size: Maximum (width, height) of the thumbnail.

        Returns:
            A numpy array containing the thumbnail image data.
        """
        pass

    @property
    @abc.abstractmethod
    def level_count(self) -> int:
        """Get the number of levels in the pyramid."""
        pass

    @property
    @abc.abstractmethod
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """Get dimensions (width, height) for each level."""
        pass

    @property
    @abc.abstractmethod
    def level_downsamples(self) -> Tuple[float, ...]:
        """Get the downsample factors for each level."""
        pass

    @property
    @abc.abstractmethod
    def properties(self) -> Dict[str, Any]:
        """Get the metadata properties of the slide."""
        pass


class NumpyMockBackend(WSIBackend):
    """A mock backend that generates synthetic slide data for testing."""

    def __init__(self) -> None:
        """Initialize the mock backend."""
        self._path: Optional[str] = None
        self._level_dims = ((10000, 10000), (5000, 5000), (2500, 2500))
        self._level_down = (1.0, 2.0, 4.0)
        self._is_open = False
        self._properties: Dict[str, Any] = {
            "mpp-x": 0.25,
            "mpp-y": 0.25,
            "objective-power": 40
        }

    def open(self, path: str) -> None:
        """Open the mock backend.

        Args:
            path: Path to the mock file.
        """
        logger.info(f"Mocking open for {path}")
        self._path = path
        self._is_open = True

    def close(self) -> None:
        """Close the mock backend."""
        logger.info(f"Mocking close for {self._path}")
        self._is_open = False
        self._path = None

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> np.ndarray:
        """Read a synthetic region from the slide.

        Args:
            location: (x, y) coordinates at level 0.
            level: The pyramid level to read from.
            size: (width, height) of the region to read.

        Returns:
            A numpy array of synthetic data.
            
        Raises:
            ValueError: If the backend is not open or arguments are invalid.
        """
        if not self._is_open:
            raise ValueError("Backend is not open")
        if level < 0 or level >= self.level_count:
            raise ValueError(f"Invalid level {level}")

        width, height = size
        # Generate some synthetic data based on location and level
        arr = np.ones((height, width, 3), dtype=np.uint8) * 240
        # Add some noise to make it look like something
        noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return arr

    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        """Get a synthetic thumbnail.

        Args:
            size: Maximum (width, height) of the thumbnail.

        Returns:
            A numpy array of thumbnail data.
            
        Raises:
            ValueError: If the backend is not open.
        """
        if not self._is_open:
            raise ValueError("Backend is not open")
        
        tw, th = size
        w, h = self.level_dimensions[0]
        scale = min(tw / w, th / h)
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        
        arr = np.ones((out_h, out_w, 3), dtype=np.uint8) * 200
        return arr

    @property
    def level_count(self) -> int:
        """Get the number of levels."""
        return len(self._level_dims)

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        """Get dimensions for each level."""
        return self._level_dims

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        """Get downsample factors."""
        return self._level_down

    @property
    def properties(self) -> Dict[str, Any]:
        """Get slide properties."""
        return self._properties
