"""Tests for CheckpointManager and FoundationCheckpoint.

Ensures that checkpoints can be correctly dumped to and loaded from JSON.
"""

import json
from typing import Any, Dict
import pytest

class FoundationCheckpoint:
    """Mock FoundationCheckpoint."""
    def __init__(self, state_dict: Dict[str, Any]) -> None:
        self.state_dict = state_dict
        
    def to_dict(self) -> Dict[str, Any]:
        return self.state_dict

class CheckpointManager:
    """Mock CheckpointManager."""
    @staticmethod
    def save(checkpoint: FoundationCheckpoint, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(checkpoint.to_dict(), f)
            
    @staticmethod
    def load(path: str) -> FoundationCheckpoint:
        with open(path, 'r') as f:
            data = json.load(f)
        return FoundationCheckpoint(data)

@pytest.fixture
def sample_states() -> list:
    return [
        {"epoch": 1, "loss": 0.5},
        {"epoch": 10, "loss": 0.1, "optimizer": "adam"},
        {"metrics": {"acc": 0.99, "f1": 0.98}, "step": 1000},
        {"model_name": "vit_b", "weights": [1, 2, 3]},
        {}
    ]

@pytest.mark.parametrize("state_index", range(5))
@pytest.mark.parametrize("checkpoint_name", ["ckpt_1.json", "best_model.json", "latest.json"])
def test_checkpoint_save_load(tmp_path: Any, sample_states: list, state_index: int, checkpoint_name: str) -> None:
    """Test saving and loading a checkpoint to/from a JSON file."""
    state_dict = sample_states[state_index]
    ckpt = FoundationCheckpoint(state_dict)
    
    file_path = tmp_path / checkpoint_name
    CheckpointManager.save(ckpt, str(file_path))
    
    loaded_ckpt = CheckpointManager.load(str(file_path))
    assert loaded_ckpt.to_dict() == state_dict

@pytest.mark.parametrize("key", ["alpha", "beta", "gamma", "delta"])
@pytest.mark.parametrize("val", [1, 3.14, "test", [1, 2, 3], {"nested": True}])
def test_checkpoint_state_types(tmp_path: Any, key: str, val: Any) -> None:
    """Test checkpointing with various data types."""
    state_dict = {key: val}
    ckpt = FoundationCheckpoint(state_dict)
    
    file_path = tmp_path / "type_test.json"
    CheckpointManager.save(ckpt, str(file_path))
    
    loaded_ckpt = CheckpointManager.load(str(file_path))
    assert loaded_ckpt.to_dict()[key] == val
