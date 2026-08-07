import logging
import uuid
from enum import Enum
from typing import Any, Dict

from bio_oncograph.deployment.config import DeploymentConfig

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Enumeration of possible deployment statuses."""
    PENDING = "PENDING"
    DEPLOYING = "DEPLOYING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    TEARDOWN = "TEARDOWN"


class DeploymentManager:
    """Manages model deployments, handling configuration and lifecycle."""

    def __init__(self, config: DeploymentConfig) -> None:
        """Initializes the DeploymentManager with a given configuration.

        Args:
            config: The deployment configuration settings.
        """
        self.config = config
        self._deployments: Dict[str, Dict[str, Any]] = {}

    def deploy(self, model_id: str, version: str) -> str:
        """Deploys a specified model version.

        Args:
            model_id: The unique identifier of the model to deploy.
            version: The semantic version of the model.

        Returns:
            The unique identifier (deployment_id) generated for this deployment.
        """
        deployment_id = str(uuid.uuid4())
        
        # Simulate deployment process based on mode
        logger.info(f"Initiating {self.config.mode} deployment for model {model_id} (v{version}).")
        
        self._deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "version": version,
            "status": DeploymentStatus.ACTIVE,
            "mode": self.config.mode,
            "host": self.config.host,
            "port": self.config.port,
        }
        
        logger.info(f"Successfully deployed model {model_id} with deployment ID {deployment_id}.")
        return deployment_id

    def teardown(self, deployment_id: str) -> bool:
        """Tears down an active deployment.

        Args:
            deployment_id: The unique identifier of the deployment to tear down.

        Returns:
            True if teardown was successful, False if the deployment was not found.
        """
        if deployment_id in self._deployments:
            self._deployments[deployment_id]["status"] = DeploymentStatus.TEARDOWN
            logger.info(f"Tearing down deployment ID {deployment_id}.")
            del self._deployments[deployment_id]
            return True
        
        logger.warning(f"Deployment ID {deployment_id} not found for teardown.")
        return False

    def get_deployment_status(self, deployment_id: str) -> DeploymentStatus:
        """Retrieves the status of a specific deployment.

        Args:
            deployment_id: The unique identifier of the deployment.

        Returns:
            The DeploymentStatus of the requested deployment.
            Returns DeploymentStatus.FAILED if not found.
        """
        deployment = self._deployments.get(deployment_id)
        if deployment:
            return deployment["status"]
        return DeploymentStatus.FAILED
