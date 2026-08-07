"""Prompt management for foundation models."""

import dataclasses
import json
import logging
from typing import Dict, List


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PromptTemplate:
    """A template for text prompts."""
    template: str
    required_kwargs: List[str]


class PromptRegistry:
    """Registry for saving, loading, and formatting prompts."""

    def __init__(self) -> None:
        """Initializes the prompt registry."""
        self._templates: Dict[str, PromptTemplate] = {}

    def register(self, name: str, template: PromptTemplate) -> None:
        """Registers a prompt template."""
        self._templates[name] = template
        logger.info(f"Registered prompt template: {name}")

    def format_prompt(self, name: str, **kwargs) -> str:
        """Formats a registered prompt with provided kwargs.

        Args:
            name: The name of the prompt template.
            **kwargs: Keyword arguments to fill the template.

        Returns:
            The formatted string.
            
        Raises:
            ValueError: If required kwargs are missing.
            KeyError: If template name is not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found.")
        
        template_obj = self._templates[name]
        for kwarg in template_obj.required_kwargs:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")
        
        return template_obj.template.format(**kwargs)

    def save(self, filepath: str) -> None:
        """Saves registered templates to a JSON file."""
        data = {
            name: {"template": t.template, "required_kwargs": t.required_kwargs}
            for name, t in self._templates.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved prompt templates to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads templates from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name, details in data.items():
            self._templates[name] = PromptTemplate(**details)
        logger.info(f"Loaded prompt templates from {filepath}")
