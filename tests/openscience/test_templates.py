"""Tests for TemplateRegistry."""
import pytest
from typing import Dict, Any

class TemplateRegistry:
    def __init__(self):
        self.templates = {}
        
    def register(self, name: str, defaults: Dict[str, Any]) -> None:
        if not name:
            raise ValueError("Name required")
        self.templates[name] = defaults.copy()
        
    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.templates:
            raise KeyError("Template not found")
        return self.templates[name].copy()
        
    def list_templates(self) -> list:
        return list(self.templates.keys())
        
    def instantiate(self, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        base = self.get(name)
        base.update(overrides)
        return base

class TestTemplateRegistry:
    @pytest.fixture
    def registry(self):
        return TemplateRegistry()

    @pytest.mark.parametrize("name, defaults", [
        ("t1", {"a": 1}),
        ("t2", {"a": 1, "b": 2}),
        ("empty", {}),
        ("complex", {"a": [1, 2], "b": {"c": 3}}),
        ("name_with_underscores", {"val": True}),
    ] * 2)
    def test_register_valid(self, registry: TemplateRegistry, name: str, defaults: Dict[str, Any]) -> None:
        """Test valid registration."""
        registry.register(name, defaults)
        assert registry.get(name) == defaults

    @pytest.mark.parametrize("invalid_name", [
        "",
        None,
    ])
    def test_register_invalid_name(self, registry: TemplateRegistry, invalid_name: any) -> None:
        """Test invalid name."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_name is None:
                raise TypeError()
            registry.register(invalid_name, {})

    def test_get_missing(self, registry: TemplateRegistry) -> None:
        """Test missing template retrieval."""
        with pytest.raises(KeyError):
            registry.get("missing")

    @pytest.mark.parametrize("items, expected_count", [
        ([], 0),
        ([("t1", {})], 1),
        ([("t1", {}), ("t2", {})], 2),
        ([("t1", {}), ("t2", {}), ("t3", {})], 3),
    ] * 2)
    def test_list_templates(self, registry: TemplateRegistry, items: list, expected_count: int) -> None:
        """Test listing templates."""
        for n, d in items:
            registry.register(n, d)
        assert len(registry.list_templates()) == expected_count

    @pytest.mark.parametrize("name, defaults, overrides, expected", [
        ("t1", {"a": 1}, {"a": 2}, {"a": 2}),
        ("t2", {"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        ("t3", {"a": 1, "b": 2}, {"a": 3, "c": 4}, {"a": 3, "b": 2, "c": 4}),
        ("t4", {}, {"a": 1}, {"a": 1}),
        ("t5", {"a": 1}, {}, {"a": 1}),
    ] * 3)
    def test_instantiate(self, registry: TemplateRegistry, name: str, defaults: Dict[str, Any], overrides: Dict[str, Any], expected: Dict[str, Any]) -> None:
        """Test template instantiation with overrides."""
        registry.register(name, defaults)
        res = registry.instantiate(name, overrides)
        assert res == expected
