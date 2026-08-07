"""Tests for Checkpoint."""
import pytest
import json
import os
from typing import Dict, Any

class CheckpointManager:
    """Mock Checkpoint Manager."""
    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        
    def save(self, state: Dict[str, Any], version: int) -> None:
        """Save a checkpoint."""
        if version < 0:
            raise ValueError("Version must be >= 0")
        path = os.path.join(self.directory, f"ckpt_{version}.json")
        with open(path, "w") as f:
            json.dump(state, f)
            
    def load(self, version: int) -> Dict[str, Any]:
        """Load a checkpoint."""
        path = os.path.join(self.directory, f"ckpt_{version}.json")
        if not os.path.exists(path):
            raise FileNotFoundError("Checkpoint not found")
        with open(path, "r") as f:
            return json.load(f)

@pytest.mark.parametrize("version", [0, 1, 10, 100, 999])
@pytest.mark.parametrize("state", [{"a": 1}, {"b": [1, 2, 3]}, {"c": "test"}])
@pytest.mark.parametrize("subdir", ["dir1", "dir2"])
def test_checkpoint_save_load(tmp_path: str, version: int, state: Dict[str, Any], subdir: str) -> None:
    """Test save and load."""
    manager = CheckpointManager(os.path.join(tmp_path, subdir))
    manager.save(state, version)
    loaded = manager.load(version)
    assert loaded == state

@pytest.mark.parametrize("version", [-1, -5, -10])
def test_checkpoint_invalid_version(tmp_path: str, version: int) -> None:
    """Test invalid version."""
    manager = CheckpointManager(str(tmp_path))
    with pytest.raises(ValueError):
        manager.save({"a": 1}, version)

@pytest.mark.parametrize("version", [1, 2, 3])
def test_checkpoint_not_found(tmp_path: str, version: int) -> None:
    """Test not found."""
    manager = CheckpointManager(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        manager.load(version)
