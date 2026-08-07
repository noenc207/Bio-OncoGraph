import logging
from typing import Any

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manages the rollback of deployments to previous stable versions."""

    def __init__(self, deployment_manager: Any, model_registry: Any) -> None:
        """Initializes the RollbackManager.

        Args:
            deployment_manager: Instance of DeploymentManager.
            model_registry: Instance of ModelRegistry.
        """
        self.deployment_manager = deployment_manager
        self.model_registry = model_registry

    def trigger_rollback(self, deployment_id: str, target_version: str) -> str:
        """Triggers a rollback of a deployment to a specified target version.

        Args:
            deployment_id: The ID of the deployment to roll back.
            target_version: The semantic version string to roll back to.

        Returns:
            A new deployment_id representing the rolled-back state.

        Raises:
            ValueError: If the current deployment or target version is invalid.
        """
        # In a real system, you'd lookup the model_id associated with the deployment_id.
        # Here we simulate fetching the associated model_id.
        logger.info(f"Initiating rollback for deployment {deployment_id} to version {target_version}.")
        
        # Teardown the current faulty deployment
        success = self.deployment_manager.teardown(deployment_id)
        if not success:
            logger.warning(f"Could not teardown deployment {deployment_id} during rollback (maybe already inactive).")

        # Simulate finding the model ID to redeploy
        # Typically retrieved from deployment_manager state
        dummy_model_id = "reverted_model_id" 
        
        # Deploy the target version
        new_deployment_id = self.deployment_manager.deploy(
            model_id=dummy_model_id, 
            version=target_version
        )
        
        logger.info(f"Rollback successful. New deployment ID: {new_deployment_id}.")
        return new_deployment_id
