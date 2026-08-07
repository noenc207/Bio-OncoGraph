"""Slide reader orchestration and backend selection."""

import logging
from typing import Tuple, Dict, Any, Type, Optional

from bio_oncograph.infrastructure.wsi.backend import WSIBackend, NumpyMockBackend
from bio_oncograph.infrastructure.wsi.slide import WholeSlideImage

logger = logging.getLogger(__name__)


class SlideReader:
    """Orchestrates reading WSI files by selecting the appropriate backend."""

    def __init__(self) -> None:
        """Initialize the SlideReader."""
        self._backends: Dict[str, Type[WSIBackend]] = {
            "mock": NumpyMockBackend,
            # Additional backends like OpenSlideBackend or CuCIMBackend would be registered here
        }

    def register_backend(self, name: str, backend_class: Type[WSIBackend]) -> None:
        """Register a new backend class.

        Args:
            name: The name of the backend.
            backend_class: The WSIBackend class to register.
        """
        self._backends[name] = backend_class
        logger.info(f"Registered WSI backend: {name}")

    def open_slide(
        self, path: str, backend_name: Optional[str] = None
    ) -> WholeSlideImage:
        """Open a slide, optionally specifying the backend to use.

        Args:
            path: Path to the WSI file.
            backend_name: Name of the backend to use. If None, uses 'mock' by default
                in this basic implementation.

        Returns:
            An opened WholeSlideImage.

        Raises:
            ValueError: If the requested backend is not registered.
            Exception: If the backend fails to open the file.
        """
        if backend_name is None:
            backend_name = "mock"

        if backend_name not in self._backends:
            raise ValueError(f"Unknown backend: {backend_name}. Available: {list(self._backends.keys())}")

        backend_class = self._backends[backend_name]
        backend_instance = backend_class()
        
        slide = WholeSlideImage(path, backend_instance)
        try:
            slide.open()
            return slide
        except Exception as e:
            logger.error(f"Failed to open slide {path} with backend {backend_name}: {e}")
            raise
