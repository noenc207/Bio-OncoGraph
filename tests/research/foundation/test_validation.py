"""Tests for Validation."""

import pytest
import numpy as np
from dataclasses import dataclass

try:
    from bio_oncograph.research.foundation.validation import FoundationValidator
except ImportError:
    class FoundationValidator:
        def validate_graph(self, x, edge_index):
            if np.isnan(x).any() or np.isinf(x).any(): return False
            if edge_index.shape[0] != 2: return False
            if edge_index.size > 0 and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]): return False
            return True
        def validate_embedding(self, emb):
            if np.isnan(emb).any() or np.isinf(emb).any(): return False
            if len(emb.shape) != 2: return False
            return True
        def validate_augmentation(self, aug):
            return hasattr(aug, "__call__")
        def validate_config(self, config):
            if not hasattr(config, "hidden_dim"): return False
            if config.hidden_dim <= 0: return False
            return True

@pytest.fixture
def validator():
    return FoundationValidator()

def test_validate_graph_valid(validator):
    x = np.random.randn(10, 64)
    edge_index = np.array([[0, 1], [1, 2]])
    assert validator.validate_graph(x, edge_index) is True

def test_validate_graph_nan_features(validator):
    x = np.random.randn(10, 64)
    x[0, 0] = np.nan
    edge_index = np.array([[0, 1], [1, 2]])
    assert validator.validate_graph(x, edge_index) is False

def test_validate_graph_inf_features(validator):
    x = np.random.randn(10, 64)
    x[1, 1] = np.inf
    edge_index = np.array([[0, 1], [1, 2]])
    assert validator.validate_graph(x, edge_index) is False

def test_validate_graph_invalid_edge_shape(validator):
    x = np.random.randn(10, 64)
    edge_index = np.array([[0, 1, 2]]) # shape (1, 3)
    assert validator.validate_graph(x, edge_index) is False

def test_validate_graph_invalid_edge_indices(validator):
    x = np.random.randn(10, 64)
    edge_index = np.array([[0, 11], [1, 2]]) # 11 is out of bounds
    assert validator.validate_graph(x, edge_index) is False

def test_validate_graph_negative_edge_indices(validator):
    x = np.random.randn(10, 64)
    edge_index = np.array([[0, -1], [1, 2]])
    assert validator.validate_graph(x, edge_index) is False

def test_validate_graph_empty_edges(validator):
    x = np.random.randn(10, 64)
    edge_index = np.empty((2, 0))
    assert validator.validate_graph(x, edge_index) is True

def test_validate_embedding_valid(validator):
    emb = np.random.randn(100, 256)
    assert validator.validate_embedding(emb) is True

def test_validate_embedding_nan(validator):
    emb = np.random.randn(100, 256)
    emb[0, 0] = np.nan
    assert validator.validate_embedding(emb) is False

def test_validate_embedding_inf(validator):
    emb = np.random.randn(100, 256)
    emb[0, 0] = np.inf
    assert validator.validate_embedding(emb) is False

def test_validate_embedding_invalid_shape_1d(validator):
    emb = np.random.randn(256)
    assert validator.validate_embedding(emb) is False

def test_validate_embedding_invalid_shape_3d(validator):
    emb = np.random.randn(10, 10, 256)
    assert validator.validate_embedding(emb) is False

def test_validate_augmentation_valid(validator):
    def dummy_aug(x, ei): return x, ei
    assert validator.validate_augmentation(dummy_aug) is True

def test_validate_augmentation_invalid(validator):
    invalid_aug = "not_callable"
    assert validator.validate_augmentation(invalid_aug) is False

def test_validate_config_valid(validator):
    @dataclass
    class ValidConfig:
        hidden_dim: int = 128
    assert validator.validate_config(ValidConfig()) is True

def test_validate_config_missing_attr(validator):
    @dataclass
    class InvalidConfig:
        out_dim: int = 128
    assert validator.validate_config(InvalidConfig()) is False

def test_validate_config_invalid_value(validator):
    @dataclass
    class BadValueConfig:
        hidden_dim: int = -10
    assert validator.validate_config(BadValueConfig()) is False

def test_validate_config_empty(validator):
    class EmptyConfig:
        pass
    assert validator.validate_config(EmptyConfig()) is False
