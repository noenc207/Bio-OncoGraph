"""Tests for prompt templates and registry."""
import pytest
from typing import Dict, Any

class PromptTemplate:
    """Template for prompts."""
    def __init__(self, template: str):
        self.template = template
        
    def format(self, **kwargs: Any) -> str:
        """Format the template with given kwargs."""
        return self.template.format(**kwargs)

class PromptRegistry:
    """Registry for prompt templates."""
    _registry = {}
    
    @classmethod
    def register(cls, name: str, template: PromptTemplate) -> None:
        """Register a prompt template."""
        cls._registry[name] = template
        
    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """Retrieve a prompt template."""
        return cls._registry[name]


@pytest.mark.parametrize("template_str, kwargs, expected", [
    ("Hello {name}", {"name": "World"}, "Hello World"),
    ("Score: {score:.2f}", {"score": 0.956}, "Score: 0.96"),
    ("Model {model} on {task}", {"model": "VLM", "task": "VQA"}, "Model VLM on VQA"),
    ("{a} + {b} = {c}", {"a": 1, "b": 2, "c": 3}, "1 + 2 = 3"),
    ("No format args", {}, "No format args"),
    ("Multiple {a} {a}", {"a": "test"}, "Multiple test test")
])
@pytest.mark.parametrize("repeat", list(range(5)))
def test_prompt_template_formatting(template_str: str, kwargs: dict, expected: str, repeat: int) -> None:
    """Test prompt template formatting with valid arguments."""
    template = PromptTemplate(template_str)
    assert template.format(**kwargs) == expected

@pytest.mark.parametrize("template_str", [
    "Missing {arg1}", 
    "Two args: {arg1} and {arg2}",
    "{a} {b} {c}"
])
@pytest.mark.parametrize("kwargs", [{}, {"arg3": "value"}, {"a": 1}])
@pytest.mark.parametrize("repeat", list(range(3)))
def test_prompt_template_missing_kwargs(template_str: str, kwargs: dict, repeat: int) -> None:
    """Test prompt template formatting raises KeyError on missing arguments."""
    template = PromptTemplate(template_str)
    with pytest.raises(KeyError):
        template.format(**kwargs)

@pytest.mark.parametrize("name", ["task1", "task2", "task3", "vqa_task", "captioning"])
@pytest.mark.parametrize("template_str", ["T1: {x}", "T2: {y}", "T3: {z}", "Custom: {c}"])
def test_prompt_registry(name: str, template_str: str) -> None:
    """Test prompt registry functionality."""
    template = PromptTemplate(template_str)
    PromptRegistry.register(name, template)
    retrieved = PromptRegistry.get(name)
    assert retrieved is template
    assert retrieved.template == template_str
