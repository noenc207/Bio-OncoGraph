"""Checkpoint management for multi-modal learning engine."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import logging
import os

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MultiModalCheckpoint:
    """Represents a saved state of the multi-modal engine.

    Attributes:
        encoders_state: State dictionary of modality encoders.
        fusion_state: State dictionary of the fusion module.
        config: Configuration dictionary used by the engine.
        metadata: Additional metadata related to the checkpoint.
        version: Version string of the checkpoint format.
    """
    encoders_state: Dict[str, Any]
    fusion_state: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str

class CheckpointManager:
    """Manages saving and loading of multi-modal engine checkpoints."""

    def __init__(self, target_version: str = "1.0.0") -> None:
        """Initialize the checkpoint manager.

        Args:
            target_version: The expected version string for checkpoints.
        """
        self.logger = logging.getLogger(__name__ + ".CheckpointManager")
        self.target_version = target_version

    def save_to_json(self, checkpoint: MultiModalCheckpoint, path: str) -> None:
        """Save a checkpoint to a JSON file.

        Args:
            checkpoint: The checkpoint object to save.
            path: The file path to save the JSON checkpoint to.
        """
        state_dict = {
            "encoders_state": checkpoint.encoders_state,
            "fusion_state": checkpoint.fusion_state,
            "config": checkpoint.config,
            "metadata": checkpoint.metadata,
            "version": checkpoint.version
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            self.logger.info(f"Checkpoint saved to {path}.")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {path}: {e}")
            raise

    def load_from_json(self, path: str) -> MultiModalCheckpoint:
        """Load a checkpoint from a JSON file.

        Args:
            path: The file path to load the JSON checkpoint from.

        Returns:
            The loaded MultiModalCheckpoint object.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            self.validate_version_compatibility(state_dict.get("version", ""))
            checkpoint = MultiModalCheckpoint(
                encoders_state=state_dict.get("encoders_state", {}),
                fusion_state=state_dict.get("fusion_state", {}),
                config=state_dict.get("config", {}),
                metadata=state_dict.get("metadata", {}),
                version=state_dict.get("version", "unknown")
            )
            self.logger.info(f"Checkpoint loaded from {path}.")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {path}: {e}")
            raise

    def validate_version_compatibility(self, version: str) -> None:
        """Validate that the checkpoint version is compatible with the target version.

        Args:
            version: The version string from the checkpoint.

        Raises:
            ValueError: If the version is incompatible.
        """
        if version != self.target_version:
            self.logger.warning(
                f"Checkpoint version '{version}' differs from target '{self.target_version}'."
            )
            if not version or version.split(".")[0] != self.target_version.split(".")[0]:
                raise ValueError(f"Incompatible checkpoint version: {version}.")
