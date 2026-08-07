"""Module for tracking experiment lineage and artifacts."""
import json
import os
from typing import Dict, Any

class ArtifactTracker:
    """Tracks experiment lineage (experiment->trial->ckpt->artifact)."""
    
    def __init__(self, base_dir: str):
        """Initializes artifact tracker.
        
        Args:
            base_dir: Base directory to save lineage tracking data.
        """
        self.base_dir = base_dir
        self.lineage: Dict[str, Any] = {}
        
    def record_experiment(self, exp_id: str, metadata: Dict[str, Any]) -> None:
        """Records a new experiment.
        
        Args:
            exp_id: Experiment identifier.
            metadata: Experiment metadata.
        """
        self.lineage[exp_id] = {"metadata": metadata, "trials": {}}
        
    def record_trial(self, exp_id: str, trial_id: str, metadata: Dict[str, Any]) -> None:
        """Records a trial for an experiment.
        
        Args:
            exp_id: Experiment identifier.
            trial_id: Trial identifier.
            metadata: Trial metadata.
        """
        if exp_id in self.lineage:
            self.lineage[exp_id]["trials"][trial_id] = {"metadata": metadata, "checkpoints": {}}
            
    def record_checkpoint(self, exp_id: str, trial_id: str, ckpt_id: str, metadata: Dict[str, Any]) -> None:
        """Records a checkpoint.
        
        Args:
            exp_id: Experiment identifier.
            trial_id: Trial identifier.
            ckpt_id: Checkpoint identifier.
            metadata: Checkpoint metadata.
        """
        if exp_id in self.lineage and trial_id in self.lineage[exp_id]["trials"]:
            self.lineage[exp_id]["trials"][trial_id]["checkpoints"][ckpt_id] = {"metadata": metadata, "artifacts": []}
            
    def record_artifact(self, exp_id: str, trial_id: str, ckpt_id: str, artifact_path: str) -> None:
        """Records an artifact associated with a checkpoint.
        
        Args:
            exp_id: Experiment identifier.
            trial_id: Trial identifier.
            ckpt_id: Checkpoint identifier.
            artifact_path: Path to the artifact.
        """
        if (exp_id in self.lineage and 
            trial_id in self.lineage[exp_id]["trials"] and 
            ckpt_id in self.lineage[exp_id]["trials"][trial_id]["checkpoints"]):
            self.lineage[exp_id]["trials"][trial_id]["checkpoints"][ckpt_id]["artifacts"].append(artifact_path)
            
    def save_lineage(self) -> None:
        """Saves lineage tracking to disk in JSON format."""
        os.makedirs(self.base_dir, exist_ok=True)
        with open(os.path.join(self.base_dir, "experiment_lineage.json"), 'w') as f:
            json.dump(self.lineage, f, indent=4)
