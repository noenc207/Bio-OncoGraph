import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing machine learning models.

    This class handles the registration, retrieval, listing, and deletion
    of models, along with their associated metadata for lineage and
    reproducibility.
    """

    def __init__(self) -> None:
        """Initializes the ModelRegistry with an empty in-memory store."""
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self, model_name: str, version: str, metadata: Dict[str, Any]
    ) -> str:
        """Registers a new model or a new version of an existing model.

        Args:
            model_name: The name of the model.
            version: The semantic version string of the model.
            metadata: Additional metadata (e.g., lineage, hyperparameters).

        Returns:
            The unique identifier (model_id) generated for this registration.
        """
        model_id = str(uuid.uuid4())
        self._registry[model_id] = {
            "model_id": model_id,
            "name": model_name,
            "version": version,
            "metadata": metadata,
        }
        logger.info(f"Registered model {model_name} version {version} with ID {model_id}.")
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a model's information by its unique identifier.

        Args:
            model_id: The unique identifier of the model.

        Returns:
            A dictionary containing the model information, or None if not found.
        """
        model = self._registry.get(model_id)
        if model is None:
            logger.warning(f"Model ID {model_id} not found in registry.")
        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """Lists all registered models.

        Returns:
            A list of dictionaries, each containing information about a registered model.
        """
        return list(self._registry.values())

    def delete_model(self, model_id: str) -> bool:
        """Deletes a model from the registry.

        Args:
            model_id: The unique identifier of the model to delete.

        Returns:
            True if the model was successfully deleted, False otherwise.
        """
        if model_id in self._registry:
            del self._registry[model_id]
            logger.info(f"Deleted model ID {model_id} from registry.")
            return True
        logger.warning(f"Failed to delete model ID {model_id}: not found.")
        return False
