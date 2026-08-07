"""Tests for explainability visualization utilities."""

import numpy as np
import pytest

def format_explanation_for_visualization(explanation_map):
    if np.max(explanation_map) == np.min(explanation_map):
        return np.zeros_like(explanation_map)
    norm = (explanation_map - np.min(explanation_map)) / (np.max(explanation_map) - np.min(explanation_map))
    return norm

def test_format_explanation_shape():
    expl = np.random.rand(10, 10)
    res = format_explanation_for_visualization(expl)
    assert res.shape == expl.shape

def test_format_explanation_bounds():
    expl = np.random.rand(5, 5) * 100 - 50
    res = format_explanation_for_visualization(expl)
    assert np.min(res) >= 0.0
    assert np.max(res) <= 1.0

def test_format_explanation_constant():
    expl = np.ones((4, 4)) * 5.0
    res = format_explanation_for_visualization(expl)
    assert np.all(res == 0.0)

def test_format_explanation_stability():
    expl = np.array([1e-10, 2e-10])
    res = format_explanation_for_visualization(expl)
    assert not np.any(np.isnan(res))

def test_format_explanation_image_like():
    expl = np.random.rand(3, 224, 224)
    res = format_explanation_for_visualization(expl)
    assert np.max(res) == 1.0
    assert np.min(res) == 0.0

def test_format_explanation_negative_values():
    expl = np.array([-10.0, 0.0, 10.0])
    res = format_explanation_for_visualization(expl)
    assert np.allclose(res, [0.0, 0.5, 1.0])

def test_format_explanation_zero_center():
    expl = np.array([-1.0, 1.0])
    res = format_explanation_for_visualization(expl)
    assert np.allclose(res, [0.0, 1.0])
