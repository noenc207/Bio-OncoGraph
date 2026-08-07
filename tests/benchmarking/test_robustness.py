"""Tests for benchmark robustness evaluation."""

import pytest
import numpy as np

@pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1, 0.2, 0.5])
@pytest.mark.parametrize("perturbation_type", ["gaussian", "salt_pepper", "adversarial", "dropout", "blur"])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_robustness_evaluation(noise_level, perturbation_type, batch_size, num_channels):
    """Ensure robustness components handle varying inputs without failure."""
    def apply_perturbation(data, p_type, n_level):
        if p_type == "gaussian":
            return data + np.random.normal(0, n_level, data.shape)
        return data

    dummy_data = np.random.rand(batch_size, num_channels, 64, 64)
    perturbed = apply_perturbation(dummy_data, perturbation_type, noise_level)
    
    assert perturbed.shape == dummy_data.shape
    if perturbation_type == "gaussian":
        assert not np.array_equal(dummy_data, perturbed)
