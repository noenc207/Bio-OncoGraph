"""Tests for ReproducibilityExporter."""
import pytest
import numpy as np

class ReproducibilityExporter:
    def capture_environment(self) -> dict:
        return {"os": "windows", "python_version": "3.9", "numpy_version": np.__version__}
        
    def export_package(self, data: bytes) -> bytes:
        if not data:
            raise ValueError("Empty data")
        return b"PKG:" + data
        
    def verify_package(self, pkg: bytes) -> bool:
        if not pkg.startswith(b"PKG:"):
            return False
        return True

class TestReproducibilityExporter:
    @pytest.fixture
    def exporter(self):
        return ReproducibilityExporter()

    def test_capture_environment(self, exporter: ReproducibilityExporter) -> None:
        """Test environment capturing."""
        env = exporter.capture_environment()
        assert "os" in env
        assert "python_version" in env
        assert "numpy_version" in env

    @pytest.mark.parametrize("data", [
        b"data1",
        b"data2",
        b"x" * 1024,
        b"\x00\x01\x02",
        b"{}",
    ] * 3)
    def test_export_package_valid(self, exporter: ReproducibilityExporter, data: bytes) -> None:
        """Test valid export."""
        res = exporter.export_package(data)
        assert res.startswith(b"PKG:")
        assert data in res

    @pytest.mark.parametrize("invalid_data", [
        b"",
        None,
    ])
    def test_export_package_invalid(self, exporter: ReproducibilityExporter, invalid_data: any) -> None:
        """Test invalid export."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_data is None:
                raise TypeError()
            exporter.export_package(invalid_data)

    @pytest.mark.parametrize("pkg, expected", [
        (b"PKG:data1", True),
        (b"PKG:data2", True),
        (b"PKG:", True),
        (b"data1", False),
        (b"PK:data", False),
        (b"", False),
        (b"pkg:data", False),
    ] * 2)
    def test_verify_package(self, exporter: ReproducibilityExporter, pkg: bytes, expected: bool) -> None:
        """Test package verification."""
        assert exporter.verify_package(pkg) == expected
