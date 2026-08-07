"""Tests for Privacy."""
import pytest
import numpy as np
from typing import List

class DifferentialPrivacy:
    """Mock Differential Privacy."""
    def __init__(self, epsilon: float, sensitivity: float) -> None:
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Laplace noise."""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

class PrivacyAccountant:
    """Mock Privacy Accountant."""
    def __init__(self, budget: float) -> None:
        self.budget = budget
        self.spent = 0.0
        
    def consume(self, amount: float) -> None:
        """Consume budget."""
        if self.spent + amount > self.budget:
            raise ValueError("Budget exceeded")
        self.spent += amount

@pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("sensitivity", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("shape", [(100,), (50, 50), (10, 10, 10)])
@pytest.mark.parametrize("seed", [1, 2])
def test_dp_noise_scale(epsilon: float, sensitivity: float, shape: tuple, seed: int) -> None:
    """Test DP noise variance scales with sensitivity/epsilon."""
    np.random.seed(seed)
    dp = DifferentialPrivacy(epsilon=epsilon, sensitivity=sensitivity)
    data = np.zeros(shape)
    noisy = dp.add_noise(data)
    
    scale = sensitivity / epsilon
    expected_var = 2 * scale ** 2
    assert np.isclose(np.var(noisy), expected_var, rtol=0.5)

@pytest.mark.parametrize("budget", [10.0, 20.0, 100.0])
@pytest.mark.parametrize("consumptions", [[1.0, 2.0], [5.0, 5.0], [9.9]])
def test_accountant_valid(budget: float, consumptions: List[float]) -> None:
    """Test accountant valid consumption."""
    acc = PrivacyAccountant(budget)
    for c in consumptions:
        acc.consume(c)
    assert np.isclose(acc.spent, sum(consumptions))

@pytest.mark.parametrize("budget", [5.0, 10.0])
@pytest.mark.parametrize("consumptions", [[6.0], [3.0, 3.0]])
def test_accountant_exceed(budget: float, consumptions: List[float]) -> None:
    """Test budget exceed throws error."""
    acc = PrivacyAccountant(budget)
    with pytest.raises(ValueError):
        for c in consumptions:
            acc.consume(c)
