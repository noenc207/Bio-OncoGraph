"""Version tracking for datasets and preprocessing."""

import hashlib
import json
from typing import Any, Dict


class VersionTracker:
    """Version tracker for dataset, annotation, and preprocessing stages."""

    def __init__(self) -> None:
        """Initializes the VersionTracker."""
        self._stages: Dict[str, Dict[str, Any]] = {}

    def record_stage(self, stage_name: str, metadata: Dict[str, Any]) -> None:
        """Records metadata for a specific pipeline stage.

        Args:
            stage_name: The name of the stage (e.g., 'dataset', 'annotation', 'preprocessing').
            metadata: Metadata associated with the stage.
        """
        self._stages[stage_name] = metadata

    def generate_version_hash(self) -> str:
        """Generates a combined version hash based on all recorded stages.

        Returns:
            A SHA256 hash string representing the overall version state.
        """
        serialized = json.dumps(self._stages, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get_stage_metadata(self, stage_name: str) -> Dict[str, Any]:
        """Retrieves metadata for a specific stage.

        Args:
            stage_name: The name of the stage.

        Returns:
            The metadata dictionary for the stage.

        Raises:
            KeyError: If the stage is not recorded.
        """
        if stage_name not in self._stages:
            raise KeyError(f"Stage '{stage_name}' not recorded.")
        return self._stages[stage_name]
        
    def export_state(self, filepath: str) -> None:
        """Exports the current version tracking state to a JSON file.
        
        Args:
            filepath: Path to save the JSON file.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'hash': self.generate_version_hash(),
                'stages': self._stages
            }, f, indent=4)
