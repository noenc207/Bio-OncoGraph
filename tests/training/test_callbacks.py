"""Tests for Callbacks."""
import pytest

class EarlyStoppingCallback:
    """Mock EarlyStoppingCallback."""
    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """Mock call."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

@pytest.mark.parametrize("losses,patience,expected_stop,stop_epoch", [
    ([1.0, 0.9, 0.8, 0.7], 3, False, -1),
    ([1.0, 1.1, 1.2, 1.3], 2, True, 2),
    ([1.0, 0.9, 0.95, 0.96, 0.97], 3, True, 4),
    *([ ( [2.0] + [2.1]*i, i, True, i ) for i in range(1, 35) ])
])
def test_early_stopping(losses: list, patience: int, expected_stop: bool, stop_epoch: int) -> None:
    """Test early stopping."""
    cb = EarlyStoppingCallback(patience=patience)
    stopped = False
    for i, loss in enumerate(losses):
        if cb(loss):
            stopped = True
            assert i == stop_epoch
            break
    assert stopped == expected_stop
