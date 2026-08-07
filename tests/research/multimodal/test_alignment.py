import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List

# ==============================================================================
# MOCKS
# ==============================================================================
class FeatureAligner(ABC):
    """Abstract base class for feature alignment."""
    @abstractmethod
    def align(self, features: np.ndarray) -> np.ndarray:
        """Align features to a common space."""
        pass

class LatentSpaceAligner(FeatureAligner):
    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def align(self, features: np.ndarray) -> np.ndarray:
        if features.shape[-1] == self.target_dim:
            return features
        elif features.shape[-1] > self.target_dim:
            return features[..., :self.target_dim]
        else:
            pad_width = [(0, 0)] * (features.ndim - 1) + [(0, self.target_dim - features.shape[-1])]
            return np.pad(features, pad_width, mode='constant', constant_values=0)

class TokenAligner(FeatureAligner):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def align(self, features: np.ndarray) -> np.ndarray:
        # Assuming shape (batch, tokens, dim)
        if features.shape[1] == self.max_tokens:
            return features
        elif features.shape[1] > self.max_tokens:
            return features[:, :self.max_tokens, :]
        else:
            pad_width = [(0, 0), (0, self.max_tokens - features.shape[1]), (0, 0)]
            return np.pad(features, pad_width, mode='constant', constant_values=0)

class EmbeddingAligner(FeatureAligner):
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor

    def align(self, features: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        return (features / norm) * self.scale_factor

# ==============================================================================
# TESTS
# ==============================================================================

@pytest.mark.parametrize("target_dim, input_shape, expected_shape", [
    (128, (32, 128), (32, 128)),
    (64, (32, 128), (32, 64)),
    (256, (32, 128), (32, 256)),
    (10, (1, 5), (1, 10))
])
def test_latent_space_aligner_shapes(target_dim: int, input_shape: Tuple[int, ...], expected_shape: Tuple[int, ...]) -> None:
    """Test LatentSpaceAligner dimension adjustments."""
    aligner = LatentSpaceAligner(target_dim)
    features = np.ones(input_shape)
    aligned = aligner.align(features)
    assert aligned.shape == expected_shape

def test_latent_space_aligner_padding_values() -> None:
    """Test LatentSpaceAligner pads with zero."""
    aligner = LatentSpaceAligner(10)
    features = np.ones((2, 5))
    aligned = aligner.align(features)
    assert np.all(aligned[:, :5] == 1)
    assert np.all(aligned[:, 5:] == 0)

@pytest.mark.parametrize("max_tokens, input_shape, expected_shape", [
    (50, (16, 50, 128), (16, 50, 128)),
    (20, (16, 50, 128), (16, 20, 128)),
    (100, (16, 50, 128), (16, 100, 128)),
    (5, (1, 1, 10), (1, 5, 10))
])
def test_token_aligner_shapes(max_tokens: int, input_shape: Tuple[int, ...], expected_shape: Tuple[int, ...]) -> None:
    """Test TokenAligner sequence length adjustments."""
    aligner = TokenAligner(max_tokens)
    features = np.ones(input_shape)
    aligned = aligner.align(features)
    assert aligned.shape == expected_shape

def test_token_aligner_determinism() -> None:
    """Test TokenAligner determinism."""
    aligner = TokenAligner(30)
    features = np.random.rand(8, 20, 64)
    out1 = aligner.align(features)
    out2 = aligner.align(features)
    np.testing.assert_array_equal(out1, out2)

@pytest.mark.parametrize("scale_factor, input_shape", [
    (1.0, (10, 32)),
    (2.5, (10, 32)),
    (0.1, (5, 5, 16))
])
def test_embedding_aligner_norm(scale_factor: float, input_shape: Tuple[int, ...]) -> None:
    """Test EmbeddingAligner normalization to sphere."""
    aligner = EmbeddingAligner(scale_factor)
    features = np.random.randn(*input_shape)
    aligned = aligner.align(features)
    norms = np.linalg.norm(aligned, axis=-1)
    np.testing.assert_allclose(norms, scale_factor, atol=1e-5)

def test_embedding_aligner_zero_vectors() -> None:
    """Test EmbeddingAligner with zero vectors (should avoid division by zero)."""
    aligner = EmbeddingAligner()
    features = np.zeros((4, 10))
    aligned = aligner.align(features)
    assert np.all(np.isnan(aligned) == False)
    np.testing.assert_array_equal(aligned, np.zeros((4, 10)))

@pytest.mark.parametrize("aligner_cls, init_args", [
    (LatentSpaceAligner, (128,)),
    (TokenAligner, (50,)),
    (EmbeddingAligner, (1.0,))
])
def test_aligners_are_subclasses(aligner_cls: type, init_args: tuple) -> None:
    """Verify aligners implement FeatureAligner."""
    aligner = aligner_cls(*init_args)
    assert isinstance(aligner, FeatureAligner)

@pytest.mark.parametrize("seed", range(5))
def test_latent_space_aligner_determinism(seed: int) -> None:
    """Ensure LatentSpaceAligner is deterministic given identical inputs."""
    np.random.seed(seed)
    aligner = LatentSpaceAligner(target_dim=64)
    features = np.random.rand(10, 100)
    out1 = aligner.align(features)
    out2 = aligner.align(features)
    np.testing.assert_array_equal(out1, out2)
