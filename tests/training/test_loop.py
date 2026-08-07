"""Tests for TrainingLoop."""
import pytest
import torch
from unittest.mock import MagicMock

class TrainingLoop:
    """Mock TrainingLoop for testing."""
    def __init__(self, model: MagicMock, optimizer: MagicMock, grad_accum_steps: int = 1) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        self.steps = 0
        
    def step(self, batch: dict, batch_idx: int) -> float:
        """Mock training step."""
        loss = self.model(batch)
        loss.backward()
        if (batch_idx + 1) % self.grad_accum_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1
        return loss.item()

@pytest.fixture
def mock_components() -> tuple:
    """Provide mocked model and optimizer."""
    model = MagicMock()
    loss_tensor = torch.tensor(1.5)
    model.return_value = MagicMock(item=lambda: loss_tensor.item(), backward=MagicMock())
    optimizer = MagicMock()
    return model, optimizer

@pytest.mark.parametrize("batch_idx,grad_accum_steps,expected_steps,expected_loss", [
    (i, acc, (i + 1) // acc, 1.5) for i in range(11) for acc in [1, 2, 3, 4]
])
def test_training_loop_step(mock_components: tuple, batch_idx: int, grad_accum_steps: int, expected_steps: int, expected_loss: float) -> None:
    """Test training loop step logic and gradient accumulation."""
    model, optimizer = mock_components
    loop = TrainingLoop(model, optimizer, grad_accum_steps)
    
    # Simulate step
    loss = 0.0
    for j in range(batch_idx + 1):
        loss = loop.step({"data": torch.randn(2)}, j)
    
    assert loss == expected_loss
    assert loop.steps == expected_steps
    assert optimizer.step.call_count == expected_steps
    assert optimizer.zero_grad.call_count == expected_steps
    assert model.call_count == batch_idx + 1
