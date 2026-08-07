"""Tests for TrainingEngine."""
import pytest

class TrainingEngine:
    """Mock TrainingEngine."""
    def __init__(self, use_gpu: bool) -> None:
        self.use_gpu = use_gpu
        self.epochs_run = 0
        
    def run(self, epochs: int) -> None:
        """Run epochs."""
        self.epochs_run = epochs

@pytest.mark.parametrize("use_gpu,epochs", [
    (g, e) for g in [True, False] for e in range(1, 21)
])
def test_training_engine_run(use_gpu: bool, epochs: int) -> None:
    """Test TrainingEngine execution."""
    engine = TrainingEngine(use_gpu=use_gpu)
    engine.run(epochs)
    assert engine.epochs_run == epochs
    assert engine.use_gpu == use_gpu
