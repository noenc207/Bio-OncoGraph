"""Tests for Graph Foundation Checkpoint."""

import pytest
import json
import os
from dataclasses import dataclass, asdict

@dataclass
class FoundationCheckpoint:
    version: str
    epoch: int
    loss: float
    model_state: dict

class CheckpointManager:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save(self, checkpoint: FoundationCheckpoint, filename: str):
        path = os.path.join(self.directory, filename)
        with open(path, 'w') as f:
            json.dump(asdict(checkpoint), f)
        return path

    def load(self, filename: str) -> FoundationCheckpoint:
        path = os.path.join(self.directory, filename)
        with open(path, 'r') as f:
            data = json.load(f)
        return FoundationCheckpoint(**data)

    def validate(self, filename: str) -> bool:
        try:
            self.load(filename)
            return True
        except Exception:
            return False

@pytest.fixture
def manager(tmp_path):
    return CheckpointManager(str(tmp_path))

@pytest.fixture
def valid_checkpoint():
    return FoundationCheckpoint(version="1.0", epoch=10, loss=0.1, model_state={"layer1": [1, 2]})

def test_checkpoint_dataclass_init(valid_checkpoint):
    assert valid_checkpoint.version == "1.0"
    assert valid_checkpoint.epoch == 10
    assert valid_checkpoint.loss == 0.1

def test_checkpoint_dataclass_asdict(valid_checkpoint):
    d = asdict(valid_checkpoint)
    assert d["version"] == "1.0"
    assert d["epoch"] == 10

def test_manager_init(tmp_path):
    m = CheckpointManager(str(tmp_path / "ckpts"))
    assert os.path.exists(m.directory)

def test_manager_save(manager, valid_checkpoint):
    p = manager.save(valid_checkpoint, "ckpt.json")
    assert os.path.exists(p)

def test_manager_load(manager, valid_checkpoint):
    manager.save(valid_checkpoint, "ckpt.json")
    loaded = manager.load("ckpt.json")
    assert loaded.version == "1.0"
    assert loaded.epoch == 10

def test_manager_validate_true(manager, valid_checkpoint):
    manager.save(valid_checkpoint, "ckpt.json")
    assert manager.validate("ckpt.json") is True

def test_manager_validate_false(manager):
    assert manager.validate("nonexistent.json") is False

def test_manager_validate_corrupt(manager):
    path = os.path.join(manager.directory, "corrupt.json")
    with open(path, 'w') as f:
        f.write("{invalid json")
    assert manager.validate("corrupt.json") is False

def test_checkpoint_version_compatibility(manager, valid_checkpoint):
    manager.save(valid_checkpoint, "ckpt.json")
    loaded = manager.load("ckpt.json")
    assert loaded.version.startswith("1.")

def test_checkpoint_json_roundtrip(valid_checkpoint):
    d = json.dumps(asdict(valid_checkpoint))
    loaded = FoundationCheckpoint(**json.loads(d))
    assert loaded == valid_checkpoint

def test_manager_save_overwrite(manager, valid_checkpoint):
    manager.save(valid_checkpoint, "ckpt.json")
    valid_checkpoint.epoch = 20
    manager.save(valid_checkpoint, "ckpt.json")
    loaded = manager.load("ckpt.json")
    assert loaded.epoch == 20

def test_checkpoint_large_epoch(manager):
    ckpt = FoundationCheckpoint("1.0", 999999, 0.0, {})
    manager.save(ckpt, "large.json")
    assert manager.load("large.json").epoch == 999999

def test_checkpoint_negative_loss():
    ckpt = FoundationCheckpoint("1.0", 1, -1.0, {})
    assert ckpt.loss == -1.0

def test_manager_multiple_saves(manager, valid_checkpoint):
    manager.save(valid_checkpoint, "ckpt1.json")
    valid_checkpoint.epoch = 2
    manager.save(valid_checkpoint, "ckpt2.json")
    assert manager.load("ckpt1.json").epoch == 10
    assert manager.load("ckpt2.json").epoch == 2

def test_checkpoint_empty_model_state():
    ckpt = FoundationCheckpoint("1.0", 1, 0.5, {})
    assert len(ckpt.model_state) == 0

def test_checkpoint_complex_model_state(manager):
    state = {"l1": {"w": [1,2], "b": [0]}, "l2": [3,4]}
    ckpt = FoundationCheckpoint("1.0", 1, 0.5, state)
    manager.save(ckpt, "c.json")
    loaded = manager.load("c.json")
    assert loaded.model_state["l1"]["w"] == [1,2]

def test_checkpoint_manager_custom_dir(tmp_path):
    d = tmp_path / "custom"
    m = CheckpointManager(str(d))
    assert m.directory == str(d)

def test_validate_missing_fields(manager):
    path = os.path.join(manager.directory, "bad.json")
    with open(path, 'w') as f:
        json.dump({"version": "1.0"}, f)
    assert manager.validate("bad.json") is False

def test_manager_save_returns_path(manager, valid_checkpoint):
    p = manager.save(valid_checkpoint, "c.json")
    assert "c.json" in p

def test_manager_save_nested_dir(manager, valid_checkpoint):
    with pytest.raises(Exception):
        # Depending on implementation, might fail if subdirs aren't created.
        # We'll just test that it attempts to save.
        manager.save(valid_checkpoint, "sub/c.json")
    
def test_checkpoint_equality(valid_checkpoint):
    ckpt2 = FoundationCheckpoint("1.0", 10, 0.1, {"layer1": [1, 2]})
    assert valid_checkpoint == ckpt2

def test_checkpoint_inequality(valid_checkpoint):
    ckpt2 = FoundationCheckpoint("1.0", 11, 0.1, {"layer1": [1, 2]})
    assert valid_checkpoint != ckpt2

def test_checkpoint_type_annotations():
    assert FoundationCheckpoint.__annotations__["epoch"] == int

def test_checkpoint_version_type():
    assert FoundationCheckpoint.__annotations__["version"] == str

def test_manager_directory_attribute(manager):
    assert hasattr(manager, "directory")
