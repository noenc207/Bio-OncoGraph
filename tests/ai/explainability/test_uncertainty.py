"""Tests for uncertainty estimation."""

import numpy as np
import pytest

def calculate_epistemic_uncertainty(predictions):
    """Variance of predictions across ensemble/MC dropout."""
    return np.var(predictions, axis=0)

def calculate_aleatoric_uncertainty(probabilities):
    """Entropy of predicted probabilities."""
    epsilon = 1e-10
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=-1)

def test_epistemic_uncertainty_shape():
    preds = np.random.rand(10, 32, 5) # 10 models/samples, batch 32, 5 classes
    unc = calculate_epistemic_uncertainty(preds)
    assert unc.shape == (32, 5)

def test_epistemic_uncertainty_zero():
    preds = np.ones((5, 10, 2))
    unc = calculate_epistemic_uncertainty(preds)
    assert np.allclose(unc, 0.0)

def test_epistemic_uncertainty_stability():
    preds = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])
    unc = calculate_epistemic_uncertainty(preds)
    assert not np.any(np.isnan(unc))
    assert np.all(unc == 0.25)

def test_epistemic_uncertainty_large_variance():
    preds = np.array([np.ones((2,2)), np.zeros((2,2))])
    unc = calculate_epistemic_uncertainty(preds)
    assert np.all(unc == 0.25)

def test_aleatoric_uncertainty_shape():
    probs = np.random.rand(32, 5)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    unc = calculate_aleatoric_uncertainty(probs)
    assert unc.shape == (32,)

def test_aleatoric_uncertainty_min():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    unc = calculate_aleatoric_uncertainty(probs)
    assert np.allclose(unc, 0.0, atol=1e-5)

def test_aleatoric_uncertainty_max():
    probs = np.array([[0.5, 0.5]])
    unc = calculate_aleatoric_uncertainty(probs)
    assert np.allclose(unc, -np.log(0.5))

def test_aleatoric_uncertainty_stability():
    probs = np.zeros((5, 10))
    unc = calculate_aleatoric_uncertainty(probs)
    assert not np.any(np.isnan(unc))

def test_aleatoric_uncertainty_uniform():
    n = 10
    probs = np.ones((2, n)) / n
    unc = calculate_aleatoric_uncertainty(probs)
    assert np.allclose(unc, np.log(n))
