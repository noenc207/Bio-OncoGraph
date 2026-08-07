"""Registry and management for code examples."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Example:
    """Represents a code example."""
    name: str
    pipeline_type: str
    code: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExampleRegistry:
    """Registry to manage and retrieve code examples."""
    
    def __init__(self) -> None:
        """Initializes the ExampleRegistry."""
        self._examples: Dict[str, Example] = {}
        logger.info("ExampleRegistry initialized.")
        
    def register_example(self, name: str, pipeline_type: str, code: str, description: str) -> None:
        """Registers a new code example.
        
        Args:
            name (str): Unique name of the example.
            pipeline_type (str): Type of pipeline the example relates to.
            code (str): The code snippet.
            description (str): Description of the example.
        """
        logger.info("Registering example: %s", name)
        self._examples[name] = Example(
            name=name,
            pipeline_type=pipeline_type,
            code=code,
            description=description
        )
        
    def get_example(self, name: str) -> Example:
        """Retrieves an example by name.
        
        Args:
            name (str): The name of the example to retrieve.
            
        Returns:
            Example: The requested example.
            
        Raises:
            KeyError: If the example is not found.
        """
        if name not in self._examples:
            logger.error("Example not found: %s", name)
            raise KeyError(f"Example {name} not found in registry.")
        return self._examples[name]
        
    def list_examples(self, pipeline_type: Optional[str] = None) -> List[Example]:
        """Lists all registered examples, optionally filtered by pipeline type.
        
        Args:
            pipeline_type (Optional[str]): If provided, only returns examples of this type.
            
        Returns:
            List[Example]: A list of examples.
        """
        if pipeline_type is None:
            return list(self._examples.values())
        return [ex for ex in self._examples.values() if ex.pipeline_type == pipeline_type]
        
    def validate_examples(self) -> List[str]:
        """Validates all registered examples for completeness.
        
        Returns:
            List[str]: A list of validation error messages.
        """
        logger.info("Validating all registered examples.")
        errors = []
        for name, example in self._examples.items():
            if not example.code.strip():
                errors.append(f"Example '{name}' has empty code.")
        return errors
