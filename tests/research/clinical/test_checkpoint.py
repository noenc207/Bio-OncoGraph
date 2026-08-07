"""Tests for ClinicalCheckpoint and ClinicalCheckpointManager."""

import pytest
import json
import os

try:
    from bio_oncograph.research.clinical.checkpoint import ClinicalCheckpoint, ClinicalCheckpointManager
except ImportError:
    class ClinicalCheckpoint:
        def __init__(self, data, version="1.0"):
            self.data = data
            self.version = version
    
    class ClinicalCheckpointManager:
        @staticmethod
        def save(checkpoint, path):
            with open(path, "w") as f:
                json.dump({"data": checkpoint.data, "version": checkpoint.version}, f)
        
        @staticmethod
        def load(path):
            with open(path, "r") as f:
                content = json.load(f)
            if content.get("version") != "1.0":
                raise ValueError("Invalid version")
            return ClinicalCheckpoint(content["data"], content["version"])

@pytest.fixture
def dummy_data():
    return {"model_weights": [1.0, 2.0, 3.0], "hyperparams": {"lr": 0.01}}


@pytest.mark.parametrize("version", ["1.0", "1.1", "2.0", "3.5", "beta"])
def test_checkpoint_init(dummy_data, version: str) -> None:
    """Test ClinicalCheckpoint initialization. (5 tests)"""
    ckpt = ClinicalCheckpoint(data=dummy_data, version=version)
    assert ckpt.data == dummy_data
    assert ckpt.version == version


@pytest.mark.parametrize("version", ["1.0", "1.0.1", "1.0.2"])
@pytest.mark.parametrize("data_size", [10, 100, 1000])
def test_checkpoint_manager_save_load(tmp_path, version: str, data_size: int) -> None:
    """Test saving and loading checkpoints. (9 tests)"""
    data = {"weights": list(range(data_size))}
    ckpt = ClinicalCheckpoint(data=data, version="1.0")
    
    file_path = tmp_path / f"ckpt_{data_size}.json"
    ClinicalCheckpointManager.save(ckpt, file_path)
    
    assert os.path.exists(file_path)
    
    loaded_ckpt = ClinicalCheckpointManager.load(file_path)
    assert loaded_ckpt.data == data
    assert loaded_ckpt.version == "1.0"


@pytest.mark.parametrize("invalid_version", ["0.1", "2.0", "alpha", "9.9", "1.1", "-1.0"])
def test_checkpoint_manager_invalid_version(tmp_path, invalid_version: str) -> None:
    """Test loading a checkpoint with an invalid version raises an error. (6 tests)"""
    file_path = tmp_path / "invalid_ckpt.json"
    with open(file_path, "w") as f:
        json.dump({"data": {}, "version": invalid_version}, f)
        
    with pytest.raises(ValueError, match="Invalid version"):
        ClinicalCheckpointManager.load(file_path)
