"""Tests for MultiModalCheckpoint and CheckpointManager in Bio-OncoGraph R007.

This module contains tests for saving and loading JSON checkpoints,
and verifying version compatibility checks.
"""

import pytest
import json
import os
import tempfile
import numpy as np

class MultiModalCheckpoint:
    """Mock MultiModalCheckpoint representing a saved state."""
    def __init__(self, data=None, version="1.0.0"):
        self.data = data or {}
        self.version = version
        
    def to_dict(self):
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                serializable_data[k] = v.tolist()
            else:
                serializable_data[k] = v
                
        return {
            "version": self.version,
            "data": serializable_data
        }
        
    @classmethod
    def from_dict(cls, state_dict):
        data = state_dict.get("data", {})
        # Convert lists back to numpy arrays
        np_data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(data=np_data, version=state_dict.get("version", "1.0.0"))

class CheckpointManager:
    """Mock CheckpointManager for saving and loading."""
    def __init__(self, expected_version="1.0.0"):
        self.expected_version = expected_version
        
    def save(self, checkpoint, filepath):
        with open(filepath, 'w') as f:
            json.dump(checkpoint.to_dict(), f)
            
    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
            
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
            
        if state_dict.get("version") != self.expected_version:
            raise ValueError(f"Version mismatch. Expected {self.expected_version}, got {state_dict.get('version')}")
            
        return MultiModalCheckpoint.from_dict(state_dict)

@pytest.fixture
def temp_ckpt_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    os.remove(path)

@pytest.fixture
def manager():
    return CheckpointManager(expected_version="1.0.0")

def test_checkpoint_init():
    ckpt = MultiModalCheckpoint(data={"a": 1})
    assert ckpt.data["a"] == 1
    assert ckpt.version == "1.0.0"

def test_checkpoint_custom_version():
    ckpt = MultiModalCheckpoint(version="2.0.0")
    assert ckpt.version == "2.0.0"

def test_checkpoint_to_dict():
    ckpt = MultiModalCheckpoint(data={"arr": np.array([1, 2])})
    d = ckpt.to_dict()
    assert d["version"] == "1.0.0"
    assert d["data"]["arr"] == [1, 2]

def test_checkpoint_from_dict():
    d = {"version": "1.5.0", "data": {"arr": [3, 4]}}
    ckpt = MultiModalCheckpoint.from_dict(d)
    assert ckpt.version == "1.5.0"
    assert isinstance(ckpt.data["arr"], np.ndarray)
    np.testing.assert_array_equal(ckpt.data["arr"], np.array([3, 4]))

def test_manager_init():
    mgr = CheckpointManager("2.0.0")
    assert mgr.expected_version == "2.0.0"

def test_manager_save_load(manager, temp_ckpt_path):
    ckpt = MultiModalCheckpoint(data={"features": np.array([0.5, 0.6])})
    manager.save(ckpt, temp_ckpt_path)
    
    loaded_ckpt = manager.load(temp_ckpt_path)
    assert loaded_ckpt.version == "1.0.0"
    np.testing.assert_array_equal(loaded_ckpt.data["features"], np.array([0.5, 0.6]))

def test_manager_load_not_found(manager):
    with pytest.raises(FileNotFoundError):
        manager.load("nonexistent_file.json")

def test_manager_version_mismatch(manager, temp_ckpt_path):
    ckpt = MultiModalCheckpoint(version="0.9.0")
    # bypass save check if any
    with open(temp_ckpt_path, 'w') as f:
        json.dump(ckpt.to_dict(), f)
        
    with pytest.raises(ValueError, match="Version mismatch"):
        manager.load(temp_ckpt_path)

def test_manager_save_empty_data(manager, temp_ckpt_path):
    ckpt = MultiModalCheckpoint()
    manager.save(ckpt, temp_ckpt_path)
    loaded = manager.load(temp_ckpt_path)
    assert len(loaded.data) == 0

def test_checkpoint_to_dict_scalar():
    ckpt = MultiModalCheckpoint(data={"val": 42})
    d = ckpt.to_dict()
    assert d["data"]["val"] == 42

def test_checkpoint_from_dict_scalar():
    d = {"version": "1.0.0", "data": {"val": 42}}
    ckpt = MultiModalCheckpoint.from_dict(d)
    assert ckpt.data["val"] == 42

def test_checkpoint_to_dict_mixed():
    ckpt = MultiModalCheckpoint(data={"arr": np.array([1]), "val": 2})
    d = ckpt.to_dict()
    assert d["data"]["arr"] == [1]
    assert d["data"]["val"] == 2

def test_checkpoint_from_dict_mixed():
    d = {"version": "1.0.0", "data": {"arr": [1], "val": 2}}
    ckpt = MultiModalCheckpoint.from_dict(d)
    assert isinstance(ckpt.data["arr"], np.ndarray)
    assert ckpt.data["val"] == 2

def test_checkpoint_multidimensional_array():
    ckpt = MultiModalCheckpoint(data={"matrix": np.zeros((2, 2))})
    d = ckpt.to_dict()
    assert d["data"]["matrix"] == [[0.0, 0.0], [0.0, 0.0]]

def test_manager_save_load_multidimensional(manager, temp_ckpt_path):
    original_matrix = np.eye(3)
    ckpt = MultiModalCheckpoint(data={"matrix": original_matrix})
    manager.save(ckpt, temp_ckpt_path)
    
    loaded = manager.load(temp_ckpt_path)
    np.testing.assert_array_equal(loaded.data["matrix"], original_matrix)

def test_manager_invalid_json(manager, temp_ckpt_path):
    with open(temp_ckpt_path, 'w') as f:
        f.write("{invalid json")
        
    with pytest.raises(json.JSONDecodeError):
        manager.load(temp_ckpt_path)

def test_checkpoint_from_dict_missing_version():
    d = {"data": {}}
    ckpt = MultiModalCheckpoint.from_dict(d)
    assert ckpt.version == "1.0.0" # default fallback

def test_manager_load_missing_version(manager, temp_ckpt_path):
    with open(temp_ckpt_path, 'w') as f:
        json.dump({"data": {}}, f)
        
    # Since default is 1.0.0 and manager expects 1.0.0, it should not raise mismatch
    # BUT wait, the state_dict.get("version") returns None, which != "1.0.0"
    with pytest.raises(ValueError, match="Version mismatch"):
        manager.load(temp_ckpt_path)

def test_checkpoint_to_dict_empty():
    ckpt = MultiModalCheckpoint()
    d = ckpt.to_dict()
    assert d == {"version": "1.0.0", "data": {}}

def test_checkpoint_from_dict_empty():
    ckpt = MultiModalCheckpoint.from_dict({})
    assert ckpt.data == {}
    assert ckpt.version == "1.0.0"

def test_manager_save_overwrite(manager, temp_ckpt_path):
    ckpt1 = MultiModalCheckpoint(data={"val": 1})
    manager.save(ckpt1, temp_ckpt_path)
    
    ckpt2 = MultiModalCheckpoint(data={"val": 2})
    manager.save(ckpt2, temp_ckpt_path)
    
    loaded = manager.load(temp_ckpt_path)
    assert loaded.data["val"] == 2
