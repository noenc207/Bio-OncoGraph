"""Tutorial management and validation utilities."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TutorialStep:
    """A single step in a tutorial."""
    title: str
    code: str
    description: str
    expected_output: str

@dataclass(frozen=True)
class Tutorial:
    """A complete tutorial with multiple steps."""
    title: str
    steps: List[TutorialStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ValidationResult:
    """Result of tutorial validation."""
    is_valid: bool
    errors: List[str]

class TutorialBuilder:
    """Builder and validator for open science tutorials."""
    
    def create_tutorial(self, title: str, steps: List[TutorialStep]) -> Tutorial:
        """Creates a new tutorial.
        
        Args:
            title (str): Tutorial title.
            steps (List[TutorialStep]): List of steps.
            
        Returns:
            Tutorial: The constructed tutorial object.
        """
        logger.info("Creating tutorial: %s", title)
        return Tutorial(title=title, steps=steps)
        
    def validate_tutorial(self, tutorial: Tutorial) -> ValidationResult:
        """Validates the structure and content of a tutorial.
        
        Args:
            tutorial (Tutorial): The tutorial to validate.
            
        Returns:
            ValidationResult: The result of the validation.
        """
        logger.info("Validating tutorial: %s", tutorial.title)
        errors = []
        if not tutorial.steps:
            errors.append("Tutorial has no steps.")
        for i, step in enumerate(tutorial.steps):
            if not step.title or not step.code:
                errors.append(f"Step {i} is missing title or code.")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
