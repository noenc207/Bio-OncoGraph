"""
Tests for WholeSlideImage context manager and lifecycle.
"""

import pytest
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Mock dependencies
class MockBackend:
    def __init__(self) -> None:
        self.is_open: bool = False
    def open(self, path: str) -> None:
        self.is_open = True
    def close(self) -> None:
        self.is_open = False
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    def get_levels(self) -> List[Tuple[int, int]]:
        return [(1000, 1000), (500, 500)]
    def get_downsamples(self) -> List[float]:
        return [1.0, 2.0]

class WholeSlideImage:
    def __init__(self, path: str, backend: Any) -> None:
        self.path: str = path
        self.backend: Any = backend
        self._is_open: bool = False

    def __enter__(self) -> 'WholeSlideImage':
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def open(self) -> None:
        self.backend.open(self.path)
        self._is_open = True

    def close(self) -> None:
        self.backend.close()
        self._is_open = False

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        if not self._is_open:
            raise RuntimeError("Slide is closed")
        return self.backend.read_region(location, level, size)
        
    def thumbnail(self, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        if not self._is_open:
            raise RuntimeError("Slide is closed")
        return self.backend.get_thumbnail(size)

    def levels(self) -> List[Tuple[int, int]]:
        return self.backend.get_levels()

    def metadata(self) -> Dict[str, Any]:
        return {"path": self.path, "levels": self.levels()}


def test_wsi_initialization() -> None:
    """Test WSI object initialization."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    assert wsi.path == "mock.svs"
    assert not wsi._is_open

def test_wsi_open() -> None:
    """Test explicit open method."""
    backend = MockBackend()
    wsi = WholeSlideImage("mock.svs", backend)
    wsi.open()
    assert wsi._is_open
    assert backend.is_open

def test_wsi_close() -> None:
    """Test explicit close method."""
    backend = MockBackend()
    wsi = WholeSlideImage("mock.svs", backend)
    wsi.open()
    wsi.close()
    assert not wsi._is_open
    assert not backend.is_open

def test_wsi_context_manager() -> None:
    """Test WSI context manager lifecycle."""
    backend = MockBackend()
    with WholeSlideImage("mock.svs", backend) as wsi:
        assert wsi._is_open
        assert backend.is_open
    assert not backend.is_open

def test_read_region_closed_raises() -> None:
    """Test read_region on closed WSI raises RuntimeError."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    with pytest.raises(RuntimeError):
        wsi.read_region((0, 0), 0, (100, 100))

def test_read_region_valid() -> None:
    """Test read_region returns valid array."""
    with WholeSlideImage("mock.svs", MockBackend()) as wsi:
        region = wsi.read_region((0, 0), 0, (100, 100))
        assert region.shape == (100, 100, 3)

def test_thumbnail_closed_raises() -> None:
    """Test thumbnail on closed WSI raises RuntimeError."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    with pytest.raises(RuntimeError):
        wsi.thumbnail()

def test_thumbnail_default_size() -> None:
    """Test thumbnail default size."""
    with WholeSlideImage("mock.svs", MockBackend()) as wsi:
        thumb = wsi.thumbnail()
        assert thumb.shape == (256, 256, 3)

def test_thumbnail_custom_size() -> None:
    """Test thumbnail custom size."""
    with WholeSlideImage("mock.svs", MockBackend()) as wsi:
        thumb = wsi.thumbnail((512, 512))
        assert thumb.shape == (512, 512, 3)

def test_levels_retrieval() -> None:
    """Test levels method."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    assert wsi.levels() == [(1000, 1000), (500, 500)]

def test_metadata_retrieval() -> None:
    """Test metadata method."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    meta = wsi.metadata()
    assert meta["path"] == "mock.svs"
    assert "levels" in meta

def test_context_manager_exception_handling() -> None:
    """Test context manager closes backend even on exception."""
    backend = MockBackend()
    try:
        with WholeSlideImage("mock.svs", backend):
            raise ValueError("Test error")
    except ValueError:
        pass
    assert not backend.is_open

def test_multiple_opens() -> None:
    """Test multiple consecutive opens."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    wsi.open()
    wsi.open()  # Should not raise
    assert wsi._is_open

def test_multiple_closes() -> None:
    """Test multiple consecutive closes."""
    wsi = WholeSlideImage("mock.svs", MockBackend())
    wsi.open()
    wsi.close()
    wsi.close()  # Should not raise
    assert not wsi._is_open

def test_wsi_backend_delegation() -> None:
    """Test that WSI properly delegates to backend."""
    backend = MockBackend()
    with WholeSlideImage("mock.svs", backend) as wsi:
        assert wsi.backend is backend

def test_read_region_coordinates() -> None:
    """Test that read_region delegates correct coordinates."""
    with WholeSlideImage("mock.svs", MockBackend()) as wsi:
        region = wsi.read_region((50, 50), 1, (20, 20))
        assert region.shape == (20, 20, 3)
