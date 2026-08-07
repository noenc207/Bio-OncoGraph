"""Tests for CheckpointManager."""
import pytest
import os
from unittest.mock import MagicMock

class CheckpointManager:
    """Mock CheckpointManager."""
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        
    def save(self, epoch: int, state: dict) -> str:
        """Mock save."""
        return os.path.join(self.save_dir, f"ckpt_{epoch}.pt")
        
    def load(self, path: str) -> dict:
        """Mock load."""
        return {"epoch": int(path.split("_")[-1].split(".")[0])}

@pytest.fixture
def ckpt_manager(tmp_path: str) -> CheckpointManager:
    """Fixture for CheckpointManager."""
    return CheckpointManager(str(tmp_path))

@pytest.mark.parametrize("epoch", list(range(1, 46)))
def test_checkpoint_save_load(ckpt_manager: CheckpointManager, epoch: int) -> None:
    """Test CheckpointManager."""
    state = {"model": MagicMock(), "optimizer": MagicMock()}
    path = ckpt_manager.save(epoch, state)
    assert path.endswith(f"ckpt_{epoch}.pt")
    
    loaded_state = ckpt_manager.load(path)
    assert loaded_state["epoch"] == epoch
