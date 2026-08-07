"""
Tests for WSI backend implementations and Abstract Base Class enforcement.
"""

import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional

# Mock definitions for testing
class WSIBackend(ABC):
    @abstractmethod
    def open(self, path: str) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
        
    @abstractmethod
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        pass
        
    @abstractmethod
    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        pass
        
    @abstractmethod
    def get_levels(self) -> List[Tuple[int, int]]:
        pass
        
    @abstractmethod
    def get_downsamples(self) -> List[float]:
        pass

class NumpyMockBackend(WSIBackend):
    def __init__(self) -> None:
        self.is_open: bool = False
        self._levels: List[Tuple[int, int]] = [(1000, 1000), (500, 500), (250, 250)]
        self._downsamples: List[float] = [1.0, 2.0, 4.0]
        
    def open(self, path: str) -> None:
        self.is_open = True
        
    def close(self) -> None:
        self.is_open = False
        
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        if not self.is_open:
            raise ValueError("Backend not open")
        if level >= len(self._levels):
            raise ValueError("Invalid level")
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
    def get_thumbnail(self, size: Tuple[int, int]) -> np.ndarray:
        if not self.is_open:
            raise ValueError("Backend not open")
        return np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        
    def get_levels(self) -> List[Tuple[int, int]]:
        return self._levels
        
    def get_downsamples(self) -> List[float]:
        return self._downsamples

# Tests
def test_backend_abc_cannot_instantiate() -> None:
    """Test that WSIBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        WSIBackend()

def test_incomplete_backend_implementation() -> None:
    """Test that incomplete backend implementations fail to instantiate."""
    class IncompleteBackend(WSIBackend):
        def open(self, path: str) -> None:
            pass
    with pytest.raises(TypeError):
        IncompleteBackend()

def test_numpy_mock_backend_instantiation() -> None:
    """Test successful instantiation of NumpyMockBackend."""
    backend = NumpyMockBackend()
    assert backend is not None

def test_numpy_mock_backend_open() -> None:
    """Test opening a slide with NumpyMockBackend."""
    backend = NumpyMockBackend()
    backend.open("mock_path.svs")
    assert backend.is_open is True

def test_numpy_mock_backend_close() -> None:
    """Test closing a slide with NumpyMockBackend."""
    backend = NumpyMockBackend()
    backend.open("mock_path.svs")
    backend.close()
    assert backend.is_open is False

def test_read_region_not_open() -> None:
    """Test reading a region when backend is closed raises error."""
    backend = NumpyMockBackend()
    with pytest.raises(ValueError, match="Backend not open"):
        backend.read_region((0, 0), 0, (100, 100))

def test_read_region_valid() -> None:
    """Test reading a valid region."""
    backend = NumpyMockBackend()
    backend.open("mock_path.svs")
    region = backend.read_region((0, 0), 0, (100, 100))
    assert isinstance(region, np.ndarray)
    assert region.shape == (100, 100, 3)

def test_read_region_invalid_level() -> None:
    """Test reading from an invalid level raises error."""
    backend = NumpyMockBackend()
    backend.open("mock_path.svs")
    with pytest.raises(ValueError, match="Invalid level"):
        backend.read_region((0, 0), 5, (100, 100))

def test_thumbnail_not_open() -> None:
    """Test thumbnail generation when backend is closed raises error."""
    backend = NumpyMockBackend()
    with pytest.raises(ValueError, match="Backend not open"):
        backend.get_thumbnail((50, 50))

def test_thumbnail_valid() -> None:
    """Test generating a valid thumbnail."""
    backend = NumpyMockBackend()
    backend.open("mock_path.svs")
    thumb = backend.get_thumbnail((50, 50))
    assert isinstance(thumb, np.ndarray)
    assert thumb.shape == (50, 50, 3)
    assert np.all(thumb == 255)

def test_get_levels() -> None:
    """Test retrieving level dimensions."""
    backend = NumpyMockBackend()
    levels = backend.get_levels()
    assert len(levels) == 3
    assert levels[0] == (1000, 1000)

def test_get_downsamples() -> None:
    """Test retrieving level downsamples."""
    backend = NumpyMockBackend()
    downsamples = backend.get_downsamples()
    assert len(downsamples) == 3
    assert downsamples[1] == 2.0

def test_read_region_type_check() -> None:
    """Test read region returns uint8."""
    backend = NumpyMockBackend()
    backend.open("mock")
    region = backend.read_region((10, 10), 0, (25, 25))
    assert region.dtype == np.uint8

def test_read_region_different_size() -> None:
    """Test read region with non-square size."""
    backend = NumpyMockBackend()
    backend.open("mock")
    region = backend.read_region((0, 0), 1, (200, 100))
    assert region.shape == (100, 200, 3)

def test_thumbnail_different_size() -> None:
    """Test thumbnail with non-square size."""
    backend = NumpyMockBackend()
    backend.open("mock")
    thumb = backend.get_thumbnail((200, 100))
    assert thumb.shape == (100, 200, 3)

def test_mock_backend_reopen() -> None:
    """Test reopening backend after close."""
    backend = NumpyMockBackend()
    backend.open("mock1")
    backend.close()
    backend.open("mock2")
    assert backend.is_open is True
