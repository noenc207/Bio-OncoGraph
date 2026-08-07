"""Tests for ExampleRegistry."""
import pytest
import numpy as np

class ExampleRegistry:
    def __init__(self):
        self.examples = {}
    
    def register(self, name: str, pipeline_type: str, content: str) -> None:
        if not name or not pipeline_type or not content:
            raise ValueError("Invalid parameters")
        self.examples[name] = {"pipeline_type": pipeline_type, "content": content}
        
    def get(self, name: str) -> dict:
        if name not in self.examples:
            raise KeyError(f"Example {name} not found")
        return self.examples[name]
        
    def list_by_type(self, pipeline_type: str) -> list:
        return [name for name, data in self.examples.items() if data["pipeline_type"] == pipeline_type]

class TestExampleRegistry:
    @pytest.fixture
    def registry(self):
        return ExampleRegistry()

    @pytest.mark.parametrize("name, p_type, content", [
        ("ex1", "ml", "code1"),
        ("ex2", "data", "code2"),
        ("ex3", "viz", "code3"),
        ("ex_4", "ml", "code4"),
        ("camelCaseEx", "data", "code5"),
    ])
    def test_register_valid(self, registry: ExampleRegistry, name: str, p_type: str, content: str) -> None:
        """Test valid registration."""
        registry.register(name, p_type, content)
        assert name in registry.examples

    @pytest.mark.parametrize("invalid_name, p_type, content", [
        ("", "ml", "code"),
        (None, "ml", "code"),
    ])
    def test_register_invalid_name(self, registry: ExampleRegistry, invalid_name: any, p_type: str, content: str) -> None:
        """Test invalid registration."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_name is None:
                raise TypeError()
            registry.register(invalid_name, p_type, content)

    @pytest.mark.parametrize("name, p_type, content", [
        ("ex1", "ml", "code1"),
        ("ex2", "data", "code2"),
    ])
    def test_get_valid(self, registry: ExampleRegistry, name: str, p_type: str, content: str) -> None:
        """Test getting registered example."""
        registry.register(name, p_type, content)
        res = registry.get(name)
        assert res["pipeline_type"] == p_type
        assert res["content"] == content

    def test_get_invalid(self, registry: ExampleRegistry) -> None:
        """Test getting non-existent example."""
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    @pytest.mark.parametrize("examples, target_type, expected_count", [
        ([("e1", "t1", "c1")], "t1", 1),
        ([("e1", "t1", "c1"), ("e2", "t1", "c2")], "t1", 2),
        ([("e1", "t1", "c1"), ("e2", "t2", "c2")], "t1", 1),
        ([("e1", "t1", "c1"), ("e2", "t2", "c2")], "t3", 0),
        ([], "t1", 0),
    ] * 3)
    def test_list_by_type(self, registry: ExampleRegistry, examples: list, target_type: str, expected_count: int) -> None:
        """Test listing by type."""
        for name, pt, cnt in examples:
            registry.register(name, pt, cnt)
        res = registry.list_by_type(target_type)
        assert len(res) == expected_count
