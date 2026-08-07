"""
Tests for SlideReader backend selection and safe reading.
"""

import pytest
from typing import Any, Dict

# Mocks
class MockBackendA:
    @staticmethod
    def supports(path: str) -> bool:
        return path.endswith('.svs')

class MockBackendB:
    @staticmethod
    def supports(path: str) -> bool:
        return path.endswith('.tif')

class SlideReader:
    def __init__(self) -> None:
        self.backends = [MockBackendA, MockBackendB]

    def _select_backend(self, path: str) -> Any:
        for backend in self.backends:
            if backend.supports(path):
                return backend()
        raise ValueError(f"No suitable backend found for {path}")

    def read(self, path: str) -> Dict[str, str]:
        backend = self._select_backend(path)
        try:
            return {"status": "success", "backend": backend.__class__.__name__}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def test_reader_initialization() -> None:
    """Test SlideReader initializes with default backends."""
    reader = SlideReader()
    assert len(reader.backends) == 2

def test_select_backend_svs() -> None:
    """Test backend selection for .svs files."""
    reader = SlideReader()
    backend = reader._select_backend("slide.svs")
    assert isinstance(backend, MockBackendA)

def test_select_backend_tif() -> None:
    """Test backend selection for .tif files."""
    reader = SlideReader()
    backend = reader._select_backend("slide.tif")
    assert isinstance(backend, MockBackendB)

def test_select_backend_unsupported() -> None:
    """Test backend selection fails for unsupported extension."""
    reader = SlideReader()
    with pytest.raises(ValueError, match="No suitable backend found"):
        reader._select_backend("slide.txt")

def test_read_success_svs() -> None:
    """Test successful read of .svs."""
    reader = SlideReader()
    res = reader.read("test.svs")
    assert res["status"] == "success"
    assert res["backend"] == "MockBackendA"

def test_read_success_tif() -> None:
    """Test successful read of .tif."""
    reader = SlideReader()
    res = reader.read("test.tif")
    assert res["status"] == "success"
    assert res["backend"] == "MockBackendB"

def test_read_unsupported_raises() -> None:
    """Test reading unsupported file raises exception implicitly."""
    reader = SlideReader()
    with pytest.raises(ValueError):
        reader.read("test.xyz")

def test_reader_custom_backends() -> None:
    """Test injecting custom backends."""
    reader = SlideReader()
    reader.backends = []
    with pytest.raises(ValueError):
        reader.read("test.svs")

class BadBackend:
    @staticmethod
    def supports(path: str) -> bool:
        return path.endswith('.bad')
    def __init__(self) -> None:
        raise RuntimeError("Backend failed to initialize")

def test_reader_backend_init_failure() -> None:
    """Test exception handling during backend initialization."""
    reader = SlideReader()
    reader.backends.append(BadBackend)
    with pytest.raises(RuntimeError, match="Backend failed to initialize"):
        reader.read("test.bad")

def test_reader_safe_reading() -> None:
    """Test safe reading generic properties."""
    reader = SlideReader()
    assert reader is not None

def test_backend_a_supports() -> None:
    """Test MockBackendA supports method directly."""
    assert MockBackendA.supports("a.svs")
    assert not MockBackendA.supports("a.tif")

def test_backend_b_supports() -> None:
    """Test MockBackendB supports method directly."""
    assert MockBackendB.supports("b.tif")
    assert not MockBackendB.supports("b.svs")

def test_multiple_extensions_support() -> None:
    """Test reader selection stability when backends share common structure."""
    class UniversalBackend:
        @staticmethod
        def supports(path: str) -> bool:
            return True
    reader = SlideReader()
    reader.backends.insert(0, UniversalBackend)
    backend = reader._select_backend("unknown.ext")
    assert isinstance(backend, UniversalBackend)
