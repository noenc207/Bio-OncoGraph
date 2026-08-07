import logging
from dataclasses import dataclass, field
from typing import List, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a validation process.
    
    Attributes:
        is_valid: Boolean indicating if the validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class EcosystemValidator:
    """Validates various components of the scientific ecosystem."""
    
    def validate_experiment(self, result: Any) -> ValidationResult:
        """Validates an experiment result.
        
        Args:
            result: The experiment result object or dict.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating experiment result")
        if not result:
            return ValidationResult(is_valid=False, errors=["Empty result."])
        return ValidationResult(is_valid=True)
        
    def validate_model_card(self, card: Any) -> ValidationResult:
        """Validates a model card.
        
        Args:
            card: The model card object or dict.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating model card")
        if not card:
            return ValidationResult(is_valid=False, errors=["Empty model card."])
        return ValidationResult(is_valid=True)
        
    def validate_dataset_card(self, card: Any) -> ValidationResult:
        """Validates a dataset card.
        
        Args:
            card: The dataset card object or dict.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating dataset card")
        if not card:
            return ValidationResult(is_valid=False, errors=["Empty dataset card."])
        return ValidationResult(is_valid=True)
        
    def validate_publication(self, package: Any) -> ValidationResult:
        """Validates a publication package.
        
        Args:
            package: The publication package object.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating publication package")
        if not package:
            return ValidationResult(is_valid=False, errors=["Empty publication package."])
        return ValidationResult(is_valid=True)
        
    def validate_artifact(self, package: Any) -> ValidationResult:
        """Validates an artifact package.
        
        Args:
            package: The artifact package object.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating artifact package")
        if not package:
            return ValidationResult(is_valid=False, errors=["Empty artifact package."])
        return ValidationResult(is_valid=True)
        
    def validate_reproducibility(self, manifest: Any) -> ValidationResult:
        """Validates a reproducibility manifest.
        
        Args:
            manifest: The reproducibility manifest object.
            
        Returns:
            A ValidationResult.
        """
        logger.info("Validating reproducibility manifest")
        if not manifest:
            return ValidationResult(is_valid=False, errors=["Empty reproducibility manifest."])
        return ValidationResult(is_valid=True)
