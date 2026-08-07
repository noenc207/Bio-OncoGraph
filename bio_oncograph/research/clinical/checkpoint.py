"""Checkpoint management for clinical models."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClinicalCheckpoint:
    """Frozen dataclass representing a clinical model checkpoint."""
    model_state: Dict[str, Any]
    calibration_state: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str
    timestamp: str


class ClinicalCheckpointManager:
    """Manager for loading and saving clinical checkpoints."""

    def save_checkpoint(self, checkpoint: ClinicalCheckpoint, path: str) -> None:
        """Save a checkpoint to a JSON file.
        
        Args:
            checkpoint: The checkpoint to save.
            path: Path to the JSON file.
        """
        logger.info("Saving clinical checkpoint to %s", path)
        data = {
            "model_state": checkpoint.model_state,
            "calibration_state": checkpoint.calibration_state,
            "config": checkpoint.config,
            "metadata": checkpoint.metadata,
            "version": checkpoint.version,
            "timestamp": checkpoint.timestamp,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info("Checkpoint saved successfully.")

    def load_checkpoint(self, path: str) -> ClinicalCheckpoint:
        """Load a checkpoint from a JSON file.
        
        Args:
            path: Path to the JSON file.
            
        Returns:
            The loaded checkpoint.
            
        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        logger.info("Loading clinical checkpoint from %s", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        checkpoint = ClinicalCheckpoint(
            model_state=data.get("model_state", {}),
            calibration_state=data.get("calibration_state", {}),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0"),
            timestamp=data.get("timestamp", ""),
        )
        logger.info("Checkpoint loaded successfully.")
        return checkpoint

    def validate_version(self, checkpoint: ClinicalCheckpoint, expected_version: str) -> bool:
        """Validate the version of a checkpoint.
        
        Args:
            checkpoint: The checkpoint to validate.
            expected_version: The expected version string.
            
        Returns:
            True if the versions match, False otherwise.
        """
        is_valid = checkpoint.version == expected_version
        if not is_valid:
            logger.warning(
                "Checkpoint version mismatch. Expected %s, got %s",
                expected_version,
                checkpoint.version,
            )
        return is_valid
