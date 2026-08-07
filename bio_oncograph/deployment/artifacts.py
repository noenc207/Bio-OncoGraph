import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages the saving, loading, and lineage tracking of deployment artifacts."""

    def __init__(self, base_dir: str) -> None:
        """Initializes the ArtifactManager.

        Args:
            base_dir: The base directory where artifacts will be stored.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.lineage_file = os.path.join(self.base_dir, "lineage.json")
        self._initialize_lineage()

    def _initialize_lineage(self) -> None:
        """Initializes the lineage tracking file if it does not exist."""
        if not os.path.exists(self.lineage_file):
            with open(self.lineage_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def save_artifact(self, artifact_id: str, obj: Any, metadata: Dict[str, Any]) -> str:
        """Saves an artifact and records its lineage.

        Args:
            artifact_id: A unique identifier for the artifact.
            obj: The Python object to serialize and save.
            metadata: Lineage and provenance metadata for the artifact.

        Returns:
            The file path where the artifact was saved.
        """
        artifact_path = os.path.join(self.base_dir, f"{artifact_id}.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump(obj, f)
        
        self.track_lineage(artifact_id, artifact_path, metadata)
        logger.info(f"Saved artifact {artifact_id} to {artifact_path}.")
        return artifact_path

    def load_artifact(self, artifact_id: str) -> Any:
        """Loads an artifact from disk.

        Args:
            artifact_id: The unique identifier for the artifact.

        Returns:
            The deserialized Python object.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
        """
        artifact_path = os.path.join(self.base_dir, f"{artifact_id}.pkl")
        if not os.path.exists(artifact_path):
            msg = f"Artifact {artifact_id} not found at {artifact_path}."
            logger.error(msg)
            raise FileNotFoundError(msg)

        with open(artifact_path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded artifact {artifact_id} from {artifact_path}.")
        return obj

    def track_lineage(self, artifact_id: str, artifact_path: str, metadata: Dict[str, Any]) -> None:
        """Records lineage metadata for a specific artifact.

        Args:
            artifact_id: The unique identifier for the artifact.
            artifact_path: The file path to the saved artifact.
            metadata: Lineage metadata dictionary.
        """
        with open(self.lineage_file, "r", encoding="utf-8") as f:
            lineage = json.load(f)

        lineage[artifact_id] = {
            "path": artifact_path,
            "metadata": metadata,
        }

        with open(self.lineage_file, "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=4)
        logger.debug(f"Tracked lineage for artifact {artifact_id}.")
