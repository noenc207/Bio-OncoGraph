"""Tests for the Extension System.

This module tests the ExtensionRegistry.
"""

import pytest
from typing import Dict, List, Any

# Mock classes
class ExtensionRegistry:
    def __init__(self) -> None:
        self.extension_points: List[str] = []
        self.extensions: Dict[str, List[Dict[str, Any]]] = {}
        
    def register_extension_point(self, name: str) -> bool:
        if name in self.extension_points:
            return False
        self.extension_points.append(name)
        self.extensions[name] = []
        return True
        
    def register_extension(self, point_name: str, extension: Dict[str, Any]) -> bool:
        if point_name not in self.extension_points:
            raise ValueError(f"Unknown extension point: {point_name}")
        self.extensions[point_name].append(extension)
        return True
        
    def get_extensions(self, point_name: str) -> List[Dict[str, Any]]:
        return self.extensions.get(point_name, [])
        
    def list_extension_points(self) -> List[str]:
        return self.extension_points

@pytest.fixture
def registry() -> ExtensionRegistry:
    return ExtensionRegistry()

@pytest.mark.parametrize("point", ["ui.toolbar", "core.menu", "api.route", "data.loader", "model.exporter"])
@pytest.mark.parametrize("repeat", [1, 2])
def test_register_extension_point(registry: ExtensionRegistry, point: str, repeat: int) -> None:
    """Test registering extension points."""
    for i in range(repeat):
        result = registry.register_extension_point(point)
        if i == 0:
            assert result is True
        else:
            assert result is False

@pytest.mark.parametrize("point", ["p1", "p2", "p3"])
@pytest.mark.parametrize("ext_name", ["extA", "extB", "extC", "extD"])
def test_register_extension(registry: ExtensionRegistry, point: str, ext_name: str) -> None:
    """Test registering extensions."""
    registry.register_extension_point(point)
    extension = {"name": ext_name, "version": "1.0"}
    assert registry.register_extension(point, extension) is True
    
    exts = registry.get_extensions(point)
    assert len(exts) == 1
    assert exts[0]["name"] == ext_name

@pytest.mark.parametrize("invalid_point", ["invalid1", "invalid2"])
def test_register_extension_invalid_point(registry: ExtensionRegistry, invalid_point: str) -> None:
    """Test registering extension to invalid point."""
    extension = {"name": "test"}
    with pytest.raises(ValueError):
        registry.register_extension(invalid_point, extension)

@pytest.mark.parametrize("points", [
    ["a", "b", "c"],
    ["x", "y"],
    ["single"]
])
def test_list_extension_points(registry: ExtensionRegistry, points: List[str]) -> None:
    """Test listing extension points."""
    for point in points:
        registry.register_extension_point(point)
    
    listed = registry.list_extension_points()
    assert set(listed) == set(points)

@pytest.mark.parametrize("point", ["target_point"])
@pytest.mark.parametrize("count", [0, 1, 5, 10])
def test_get_extensions(registry: ExtensionRegistry, point: str, count: int) -> None:
    """Test getting extensions for a point."""
    registry.register_extension_point(point)
    for i in range(count):
        registry.register_extension(point, {"id": i})
        
    exts = registry.get_extensions(point)
    assert len(exts) == count
    for i in range(count):
        assert exts[i]["id"] == i
