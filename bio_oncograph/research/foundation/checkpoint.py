"""Checkpoint management for foundation models."""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import datetime

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FoundationCheckpoint:
    """Frozen dataclass representing a training checkpoint."""
    encoder_state: Dict[str, Any]
    projector_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str
    timestamp: str

class CheckpointManager:
    """Manager for saving and loading checkpoints."""

    def save(self, checkpoint: FoundationCheckpoint, path: str) -> None:
        """Serialize checkpoint to JSON.

        Args:
            checkpoint: FoundationCheckpoint instance.
            path: Path to save the JSON file.
        """
        with open(path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=4)
        logger.info("Successfully saved checkpoint to %s", path)

    def load(self, path: str) -> FoundationCheckpoint:
        """Deserialize checkpoint from JSON.

        Args:
            path: Path to the JSON checkpoint file.

        Returns:
            Loaded FoundationCheckpoint instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        checkpoint = FoundationCheckpoint(**data)
        if not self.validate(checkpoint):
            logger.warning("Checkpoint validation failed for %s", path)
            
        return checkpoint

    def validate(self, checkpoint: FoundationCheckpoint) -> bool:
        """Check version compatibility and required keys.

        Args:
            checkpoint: Checkpoint to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not hasattr(checkpoint, 'version') or checkpoint.version != "1.0":
            logger.error("Incompatible checkpoint version: %s", getattr(checkpoint, 'version', None))
            return False
            
        required_keys = ['encoder_state', 'projector_state', 'config']
        for key in required_keys:
            if not getattr(checkpoint, key):
                logger.error("Missing required checkpoint key: %s", key)
                return False
                
        return True
