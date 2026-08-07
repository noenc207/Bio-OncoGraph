"""Tests for attribution methods."""

import numpy as np
import pytest

class GradientAttribution:
    def __init__(self, model_mock=None):
        self.model = model_mock

    def attribute(self, inputs):
        # mock attribution: return constant gradients
        return np.ones_like(inputs) * 0.5

class AttentionAttribution:
    def __init__(self, num_heads=4):
        self.num_heads = num_heads

    def attribute(self, attention_weights):
        # mock: average over heads
        return np.mean(attention_weights, axis=1)

def test_gradient_attribution_shape():
    attr = GradientAttribution()
    inputs = np.random.rand(2, 10, 5)
    result = attr.attribute(inputs)
    assert result.shape == inputs.shape

def test_gradient_attribution_values():
    attr = GradientAttribution()
    inputs = np.zeros((3, 3))
    result = attr.attribute(inputs)
    assert np.allclose(result, 0.5)

def test_gradient_attribution_no_nans():
    attr = GradientAttribution()
    inputs = np.array([np.nan, np.inf, 1.0])
    result = attr.attribute(inputs)
    assert not np.any(np.isnan(result))

def test_gradient_attribution_batch():
    attr = GradientAttribution()
    inputs = np.random.rand(32, 128)
    result = attr.attribute(inputs)
    assert result.shape == (32, 128)

def test_attention_attribution_shape():
    attr = AttentionAttribution(num_heads=4)
    # batch, heads, seq_len, seq_len
    weights = np.random.rand(2, 4, 10, 10)
    result = attr.attribute(weights)
    assert result.shape == (2, 10, 10)

def test_attention_attribution_values():
    attr = AttentionAttribution()
    weights = np.ones((1, 2, 5, 5))
    weights[:, 1, :, :] = 0.5
    result = attr.attribute(weights)
    assert np.allclose(result, 0.75)

def test_attention_attribution_stability():
    attr = AttentionAttribution()
    weights = np.zeros((1, 1, 3, 3))
    result = attr.attribute(weights)
    assert not np.any(np.isnan(result))

def test_gradient_attribution_high_dim():
    attr = GradientAttribution()
    inputs = np.random.rand(2, 3, 4, 5, 6)
    result = attr.attribute(inputs)
    assert result.shape == inputs.shape

def test_attention_attribution_single_head():
    attr = AttentionAttribution(num_heads=1)
    weights = np.random.rand(5, 1, 8, 8)
    result = attr.attribute(weights)
    assert result.shape == (5, 8, 8)

def test_attention_attribution_zero_weights():
    attr = AttentionAttribution()
    weights = np.zeros((2, 4, 6, 6))
    result = attr.attribute(weights)
    assert np.all(result == 0.0)
