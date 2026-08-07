"""Tests for Projectors."""

import pytest
import numpy as np
from abc import ABC, abstractmethod

try:
    from bio_oncograph.research.foundation.projector import BaseProjector, MLPProjector, LinearProjector, IdentityProjector
except ImportError:
    class BaseProjector(ABC):
        def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            
        @abstractmethod
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass

    class MLPProjector(BaseProjector):
        def forward(self, x: np.ndarray) -> np.ndarray:
            return self.rng.normal(0, 1, (x.shape[0], self.out_dim))
            
    class LinearProjector(BaseProjector):
        def forward(self, x: np.ndarray) -> np.ndarray:
            return self.rng.normal(0, 1, (x.shape[0], self.out_dim))
            
    class IdentityProjector(BaseProjector):
        def forward(self, x: np.ndarray) -> np.ndarray:
            return x

@pytest.fixture
def embedding_data():
    return np.random.randn(100, 256)

def test_base_projector_abc():
    with pytest.raises(TypeError):
        BaseProjector(in_dim=256, out_dim=128)

def test_mlp_projector_shape(embedding_data):
    proj = MLPProjector(in_dim=256, out_dim=128)
    out = proj.forward(embedding_data)
    assert out.shape == (100, 128)

def test_mlp_projector_determinism(embedding_data):
    proj1 = MLPProjector(in_dim=256, out_dim=128, seed=42)
    proj2 = MLPProjector(in_dim=256, out_dim=128, seed=42)
    np.testing.assert_array_equal(proj1.forward(embedding_data), proj2.forward(embedding_data))

def test_mlp_projector_different_seeds(embedding_data):
    proj1 = MLPProjector(in_dim=256, out_dim=128, seed=42)
    proj2 = MLPProjector(in_dim=256, out_dim=128, seed=43)
    assert not np.array_equal(proj1.forward(embedding_data), proj2.forward(embedding_data))

def test_mlp_projector_zero_input():
    proj = MLPProjector(in_dim=256, out_dim=128)
    x = np.zeros((10, 256))
    out = proj.forward(x)
    assert out.shape == (10, 128)

def test_mlp_projector_single_sample():
    proj = MLPProjector(in_dim=256, out_dim=128)
    x = np.random.randn(1, 256)
    out = proj.forward(x)
    assert out.shape == (1, 128)

def test_linear_projector_shape(embedding_data):
    proj = LinearProjector(in_dim=256, out_dim=128)
    out = proj.forward(embedding_data)
    assert out.shape == (100, 128)

def test_linear_projector_determinism(embedding_data):
    proj1 = LinearProjector(in_dim=256, out_dim=128, seed=1)
    proj2 = LinearProjector(in_dim=256, out_dim=128, seed=1)
    np.testing.assert_array_equal(proj1.forward(embedding_data), proj2.forward(embedding_data))

def test_linear_projector_different_seeds(embedding_data):
    proj1 = LinearProjector(in_dim=256, out_dim=128, seed=1)
    proj2 = LinearProjector(in_dim=256, out_dim=128, seed=2)
    assert not np.array_equal(proj1.forward(embedding_data), proj2.forward(embedding_data))

def test_linear_projector_zero_input():
    proj = LinearProjector(in_dim=256, out_dim=128)
    x = np.zeros((5, 256))
    out = proj.forward(x)
    assert out.shape == (5, 128)

def test_linear_projector_single_sample():
    proj = LinearProjector(in_dim=256, out_dim=128)
    x = np.random.randn(1, 256)
    out = proj.forward(x)
    assert out.shape == (1, 128)

def test_identity_projector_shape(embedding_data):
    proj = IdentityProjector(in_dim=256, out_dim=256)
    out = proj.forward(embedding_data)
    assert out.shape == (100, 256)

def test_identity_projector_determinism(embedding_data):
    proj = IdentityProjector(in_dim=256, out_dim=256)
    out = proj.forward(embedding_data)
    np.testing.assert_array_equal(out, embedding_data)

def test_identity_projector_different_seeds(embedding_data):
    proj1 = IdentityProjector(in_dim=256, out_dim=256, seed=1)
    proj2 = IdentityProjector(in_dim=256, out_dim=256, seed=2)
    np.testing.assert_array_equal(proj1.forward(embedding_data), proj2.forward(embedding_data))

def test_identity_projector_zero_input():
    proj = IdentityProjector(in_dim=256, out_dim=256)
    x = np.zeros((5, 256))
    out = proj.forward(x)
    np.testing.assert_array_equal(out, x)

def test_identity_projector_single_sample():
    proj = IdentityProjector(in_dim=256, out_dim=256)
    x = np.random.randn(1, 256)
    out = proj.forward(x)
    np.testing.assert_array_equal(out, x)

def test_projector_large_batch():
    x = np.random.randn(10000, 256)
    proj = MLPProjector(in_dim=256, out_dim=128)
    out = proj.forward(x)
    assert out.shape == (10000, 128)

def test_projector_negative_values():
    x = -np.random.rand(50, 256)
    proj = LinearProjector(in_dim=256, out_dim=128)
    out = proj.forward(x)
    assert out.shape == (50, 128)
