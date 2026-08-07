import logging
import os
from dataclasses import dataclass
from typing import Any, List

from bio_oncograph.deployment.deployment import DeploymentStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a validation check.

    Attributes:
        is_valid: True if the validation passed, False otherwise.
        errors: A list of error messages encountered during validation.
    """
    is_valid: bool
    errors: List[str]


class DeploymentValidator:
    """Performs validation checks on deployment components and states."""

    def __init__(self, deployment_manager: Any, registry: Any, artifact_manager: Any) -> None:
        """Initializes the DeploymentValidator.

        Args:
            deployment_manager: Instance of DeploymentManager.
            registry: Instance of ModelRegistry.
            artifact_manager: Instance of ArtifactManager.
        """
        self.deployment_manager = deployment_manager
        self.registry = registry
        self.artifact_manager = artifact_manager

    def validate_deployment_status(self, deployment_id: str) -> ValidationResult:
        """Validates that a deployment has an ACTIVE status.

        Args:
            deployment_id: The ID of the deployment to check.

        Returns:
            A ValidationResult indicating success or failure.
        """
        errors = []
        status = self.deployment_manager.get_deployment_status(deployment_id)
        
        if status != DeploymentStatus.ACTIVE:
            msg = f"Deployment {deployment_id} is not ACTIVE. Current status: {status}."
            logger.error(msg)
            errors.append(msg)
            
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_registry_consistency(self, model_id: str) -> ValidationResult:
        """Validates that a model exists in the registry and is consistent.

        Args:
            model_id: The ID of the model to check.

        Returns:
            A ValidationResult indicating success or failure.
        """
        errors = []
        model_info = self.registry.get_model(model_id)
        
        if not model_info:
            msg = f"Model {model_id} not found in registry."
            logger.error(msg)
            errors.append(msg)
        elif "version" not in model_info:
            msg = f"Model {model_id} is missing version information."
            logger.error(msg)
            errors.append(msg)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_artifacts(self, artifact_ids: List[str]) -> ValidationResult:
        """Validates the existence of specified artifacts on disk.

        Args:
            artifact_ids: A list of artifact IDs to verify.

        Returns:
            A ValidationResult indicating success or failure.
        """
        errors = []
        base_dir = self.artifact_manager.base_dir
        
        for art_id in artifact_ids:
            path = os.path.join(base_dir, f"{art_id}.pkl")
            if not os.path.exists(path):
                msg = f"Artifact {art_id} file not found at {path}."
                logger.error(msg)
                errors.append(msg)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
