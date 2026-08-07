"""Checkpoint management for federated learning."""

import dataclasses
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

from bio_oncograph.research.federated.config import FederatedConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FederatedCheckpoint:
    """Represents a snapshot of the federated learning state.

    Attributes:
        round_num: The current communication round number.
        global_model: The global model parameters.
        client_states: States of the participating clients.
        aggregation_state: Internal state of the aggregation strategy.
        privacy_state: Internal state of the privacy mechanism/accountant.
        config: The configuration used for this session.
        version: Version identifier for the checkpoint format.
    """
    round_num: int
    global_model: np.ndarray
    client_states: Dict[str, Any]
    aggregation_state: Dict[str, Any]
    privacy_state: Dict[str, Any]
    config: FederatedConfig
    version: str = "1.0"


class CheckpointManager:
    """Manages saving and loading of federated checkpoints."""

    def __init__(self, checkpoint_dir: str) -> None:
        """Initializes the checkpoint manager.

        Args:
            checkpoint_dir: Directory where checkpoints are stored.
        """
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)
        logger.info("Checkpoint manager initialized at %s.", checkpoint_dir)

    def _array_to_list(self, obj: Any) -> Any:
        """Recursively converts numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._array_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._array_to_list(v) for v in obj]
        return obj

    def _list_to_array(self, obj: Any) -> Any:
        """Recursively attempts to convert lists to numpy arrays during loading."""
        if isinstance(obj, list):
            # Try to convert to array, but fallback if it's not a numeric array
            try:
                arr = np.array(obj)
                if arr.dtype.kind in "iufc":
                    return arr
            except Exception:
                pass
            return [self._list_to_array(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: self._list_to_array(v) for k, v in obj.items()}
        return obj

    def save(self, checkpoint: FederatedCheckpoint, filename: str) -> None:
        """Saves a checkpoint to disk.

        Args:
            checkpoint: The federated checkpoint object.
            filename: The name of the file to save the checkpoint as.
        """
        filepath = os.path.join(self._checkpoint_dir, filename)
        
        # Prepare dictionary for JSON serialization
        data = {
            "round_num": checkpoint.round_num,
            "global_model": self._array_to_list(checkpoint.global_model),
            "client_states": self._array_to_list(checkpoint.client_states),
            "aggregation_state": self._array_to_list(checkpoint.aggregation_state),
            "privacy_state": self._array_to_list(checkpoint.privacy_state),
            "config": dataclasses.asdict(checkpoint.config),
            "version": checkpoint.version
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        logger.info("Saved checkpoint to %s.", filepath)

    def load(self, filename: str) -> FederatedCheckpoint:
        """Loads a checkpoint from disk.

        Args:
            filename: The name of the file to load the checkpoint from.

        Returns:
            The loaded federated checkpoint object.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        filepath = os.path.join(self._checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error("Checkpoint file %s not found.", filepath)
            raise FileNotFoundError(f"Checkpoint {filepath} not found.")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = FederatedConfig(**data["config"])
        global_model = np.array(data["global_model"])
        
        checkpoint = FederatedCheckpoint(
            round_num=data["round_num"],
            global_model=global_model,
            client_states=self._list_to_array(data["client_states"]),
            aggregation_state=self._list_to_array(data["aggregation_state"]),
            privacy_state=self._list_to_array(data["privacy_state"]),
            config=config,
            version=data["version"]
        )
        
        logger.info("Loaded checkpoint from %s.", filepath)
        return checkpoint
