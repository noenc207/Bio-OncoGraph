"""Checkpointing module for foundation models."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundationCheckpoint:
    """Represents a saved state of a foundation model engine.

    Attributes:
        registry_state: The state of registered models.
        cache_metadata: Metadata describing the cache.
        config: Configuration dictionary.
        version: Version string of the checkpoint format.
    """
    registry_state: Dict[str, Any]
    cache_metadata: Dict[str, Any]
    config: Dict[str, Any]
    version: str


class CheckpointManager:
    """Manages saving and loading of foundation checkpoints."""

    def __init__(self) -> None:
        """Initializes the CheckpointManager."""
        logger.debug("Initializing CheckpointManager.")

    def save(self, checkpoint: FoundationCheckpoint, path: str) -> None:
        """Saves a checkpoint to a file securely.

        Args:
            checkpoint: The checkpoint to save.
            path: File path to save the JSON file.
        """
        logger.info("Saving checkpoint to %s", path)
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=4)

    def load(self, path: str) -> FoundationCheckpoint:
        """Loads a checkpoint from a file.

        Args:
            path: File path of the JSON checkpoint.

        Returns:
            The loaded FoundationCheckpoint.
        """
        logger.info("Loading checkpoint from %s", path)
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return FoundationCheckpoint(
            registry_state=data.get("registry_state", {}),
            cache_metadata=data.get("cache_metadata", {}),
            config=data.get("config", {}),
            version=data.get("version", "1.0"),
        )
