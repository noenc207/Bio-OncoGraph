"""Lineage tracking and registry for Experiments, Trials, Checkpoints, and Artifacts."""
import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Registry:
    """Tracks lineage information for the experiment lifecycle.
    
    Attributes:
        registry_path: Path to the registry log file.
    """
    
    def __init__(self, registry_dir: str = "./registry"):
        """Initializes the Registry."""
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "lineage.jsonl")
        
    def _log_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Appends a structured event to the registry log."""
        record = {
            "event_type": event_type,
            "metadata": metadata
        }
        with open(self.registry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
    def register_experiment(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """Registers a new experiment."""
        self._log_event("EXPERIMENT_CREATED", {"experiment_id": experiment_id, "config": config})
        
    def register_trial(self, experiment_id: str, trial_id: int, params: Dict[str, Any]) -> None:
        """Registers a new trial within an experiment."""
        self._log_event("TRIAL_STARTED", {
            "experiment_id": experiment_id,
            "trial_id": trial_id,
            "params": params
        })
        
    def register_checkpoint(self, trial_id: int, epoch: int, path: str) -> None:
        """Registers a checkpoint for a specific trial."""
        self._log_event("CHECKPOINT_SAVED", {
            "trial_id": trial_id,
            "epoch": epoch,
            "path": path
        })
        
    def register_artifact(self, trial_id: int, artifact_name: str, path: str) -> None:
        """Registers an artifact produced by a trial."""
        self._log_event("ARTIFACT_GENERATED", {
            "trial_id": trial_id,
            "artifact_name": artifact_name,
            "path": path
        })
