"""Experiment template registry and management."""

import logging
import copy
from dataclasses import dataclass
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ExperimentTemplate:
    """A template for creating standardized experiments."""
    name: str
    description: str
    config_schema: Dict[str, Any]
    default_config: Dict[str, Any]

class TemplateRegistry:
    """Registry for storing and instantiating experiment templates."""
    
    def __init__(self) -> None:
        """Initializes the TemplateRegistry."""
        self._templates: Dict[str, ExperimentTemplate] = {}
        logger.info("TemplateRegistry initialized.")
        
    def register_template(self, template: ExperimentTemplate) -> None:
        """Registers a new experiment template.
        
        Args:
            template (ExperimentTemplate): The template to register.
        """
        logger.info("Registering template: %s", template.name)
        self._templates[template.name] = template
        
    def get_template(self, name: str) -> ExperimentTemplate:
        """Retrieves a template by name.
        
        Args:
            name (str): The name of the template.
            
        Returns:
            ExperimentTemplate: The requested template.
            
        Raises:
            KeyError: If the template is not found.
        """
        if name not in self._templates:
            logger.error("Template not found: %s", name)
            raise KeyError(f"Template '{name}' not found.")
        return self._templates[name]
        
    def list_templates(self) -> List[ExperimentTemplate]:
        """Lists all registered templates.
        
        Returns:
            List[ExperimentTemplate]: A list of all templates.
        """
        return list(self._templates.values())
        
    def instantiate(self, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiates a template by merging default config with overrides.
        
        Args:
            name (str): The name of the template.
            overrides (Dict[str, Any]): Configuration overrides provided by the user.
            
        Returns:
            Dict[str, Any]: The fully instantiated configuration.
        """
        logger.info("Instantiating template: %s with overrides.", name)
        template = self.get_template(name)
        
        # Start with the defaults and apply overrides
        config = copy.deepcopy(template.default_config)
        config.update(overrides)
        
        return config
