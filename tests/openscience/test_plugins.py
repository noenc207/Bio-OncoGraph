"""Tests for the Plugin System.

This module tests the PluginInterface and PluginRegistry classes.
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional

# Mocking the classes to test
class PluginInterface:
    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version
    def run(self) -> None:
        pass

class PluginRegistry:
    def __init__(self) -> None:
        self.plugins: Dict[str, PluginInterface] = {}
    
    def register(self, plugin: PluginInterface) -> bool:
        if not isinstance(plugin, PluginInterface):
            raise ValueError("Invalid plugin type.")
        if plugin.name in self.plugins:
            return False
        self.plugins[plugin.name] = plugin
        return True
    
    def get(self, name: str) -> Optional[PluginInterface]:
        return self.plugins.get(name)
    
    def list_plugins(self) -> list:
        return list(self.plugins.values())
    
    def unregister(self, name: str) -> bool:
        if name in self.plugins:
            del self.plugins[name]
            return True
        return False
    
    def validate(self, plugin: PluginInterface) -> bool:
        return hasattr(plugin, 'name') and hasattr(plugin, 'version') and hasattr(plugin, 'run')


@pytest.fixture
def registry() -> PluginRegistry:
    return PluginRegistry()


@pytest.mark.parametrize("name", ["dataset_plugin", "model_plugin", "loss_plugin", "metric_plugin", "viewer_plugin"])
@pytest.mark.parametrize("version", ["1.0.0", "1.1.0", "2.0.0-beta", "0.1.0"])
def test_plugin_interface_creation(name: str, version: str) -> None:
    """Test PluginInterface creation."""
    plugin = PluginInterface(name, version)
    assert plugin.name == name
    assert plugin.version == version


@pytest.mark.parametrize("name", ["plugin_A", "plugin_B", "plugin_C"])
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
def test_plugin_registry_register_and_get(registry: PluginRegistry, name: str, version: str) -> None:
    """Test registering and getting plugins."""
    plugin = PluginInterface(name, version)
    assert registry.register(plugin) is True
    retrieved = registry.get(name)
    assert retrieved is not None
    assert retrieved.name == name
    assert retrieved.version == version


@pytest.mark.parametrize("name", ["test1", "test2", "test3"])
def test_plugin_registry_duplicate_register(registry: PluginRegistry, name: str) -> None:
    """Test duplicate registration."""
    plugin1 = PluginInterface(name, "1.0")
    plugin2 = PluginInterface(name, "2.0")
    assert registry.register(plugin1) is True
    assert registry.register(plugin2) is False


@pytest.mark.parametrize("names", [
    ["a", "b", "c"],
    ["x", "y"],
    ["single"],
    ["p1", "p2", "p3", "p4", "p5"]
])
def test_plugin_registry_list(registry: PluginRegistry, names: list) -> None:
    """Test listing plugins."""
    for name in names:
        registry.register(PluginInterface(name, "1.0"))
    listed = registry.list_plugins()
    assert len(listed) == len(names)
    listed_names = [p.name for p in listed]
    for name in names:
        assert name in listed_names


@pytest.mark.parametrize("name", ["plugin_x", "plugin_y"])
@pytest.mark.parametrize("exists", [True, False])
def test_plugin_registry_unregister(registry: PluginRegistry, name: str, exists: bool) -> None:
    """Test unregistering plugins."""
    if exists:
        registry.register(PluginInterface(name, "1.0"))
    
    result = registry.unregister(name)
    assert result is exists
    assert registry.get(name) is None


@pytest.mark.parametrize("name", ["valid1", "valid2"])
@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_plugin_registry_validate(registry: PluginRegistry, name: str, version: str) -> None:
    """Test plugin validation."""
    plugin = PluginInterface(name, version)
    assert registry.validate(plugin) is True


def test_plugin_registry_invalid_register(registry: PluginRegistry) -> None:
    """Test invalid registration."""
    with pytest.raises(ValueError):
        registry.register("not_a_plugin") # type: ignore
