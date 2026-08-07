import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

# ==============================================================================
# MOCKS
# ==============================================================================
class FusionStrategy(ABC):
    @abstractmethod
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        pass

class EarlyFusion(FusionStrategy):
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        if not modalities:
            raise ValueError("No modalities provided.")
        return np.concatenate(modalities, axis=-1)

class IntermediateFusion(FusionStrategy):
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        if not modalities:
            raise ValueError("No modalities provided.")
        # Mock intermediate fusion as element-wise mean
        shapes = [m.shape for m in modalities]
        if len(set(shapes)) > 1:
            raise ValueError("Modalities must have same shape for element-wise mean.")
        return np.mean(np.stack(modalities), axis=0)

class LateFusion(FusionStrategy):
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        if not modalities:
            raise ValueError("No modalities provided.")
        # Mock late fusion as max voting/pooling
        shapes = [m.shape for m in modalities]
        if len(set(shapes)) > 1:
            raise ValueError("Modalities must have same shape for max pooling.")
        return np.max(np.stack(modalities), axis=0)

class HierarchicalFusion(FusionStrategy):
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        if len(modalities) < 2:
            raise ValueError("Hierarchical fusion requires at least two modalities.")
        # Mock: concat first two, then mean with rest if shapes align, else just concat all
        return np.concatenate(modalities, axis=-1)

class AttentionFusion(FusionStrategy):
    def fuse(self, modalities: List[np.ndarray]) -> np.ndarray:
        if not modalities:
            raise ValueError("No modalities provided.")
        shapes = [m.shape for m in modalities]
        if len(set(shapes)) > 1:
            raise ValueError("Modalities must have same shape for attention fusion.")
        stacked = np.stack(modalities)
        # Mock attention weights
        weights = np.ones(len(modalities)) / len(modalities)
        weights_reshaped = weights.reshape(-1, *([1] * modalities[0].ndim))
        return np.sum(stacked * weights_reshaped, axis=0)

# ==============================================================================
# TESTS
# ==============================================================================

@pytest.fixture
def uniform_modalities() -> List[np.ndarray]:
    return [
        np.ones((2, 4)),
        np.ones((2, 4)) * 2,
        np.ones((2, 4)) * 3
    ]

@pytest.fixture
def variable_modalities() -> List[np.ndarray]:
    return [
        np.ones((2, 4)),
        np.ones((2, 8)),
        np.ones((2, 2))
    ]

def test_early_fusion_shapes(variable_modalities: List[np.ndarray]) -> None:
    """Test EarlyFusion concatenation."""
    fusion = EarlyFusion()
    fused = fusion.fuse(variable_modalities)
    assert fused.shape == (2, 14)

def test_intermediate_fusion_logic(uniform_modalities: List[np.ndarray]) -> None:
    """Test IntermediateFusion computes mean."""
    fusion = IntermediateFusion()
    fused = fusion.fuse(uniform_modalities)
    np.testing.assert_array_equal(fused, np.ones((2, 4)) * 2.0)

def test_intermediate_fusion_shape_error(variable_modalities: List[np.ndarray]) -> None:
    """Test IntermediateFusion raises error on shape mismatch."""
    fusion = IntermediateFusion()
    with pytest.raises(ValueError, match="same shape"):
        fusion.fuse(variable_modalities)

def test_late_fusion_logic(uniform_modalities: List[np.ndarray]) -> None:
    """Test LateFusion computes max."""
    fusion = LateFusion()
    fused = fusion.fuse(uniform_modalities)
    np.testing.assert_array_equal(fused, np.ones((2, 4)) * 3.0)

def test_hierarchical_fusion_min_modalities() -> None:
    """Test HierarchicalFusion requires >= 2 modalities."""
    fusion = HierarchicalFusion()
    with pytest.raises(ValueError, match="at least two"):
        fusion.fuse([np.ones((2, 2))])

def test_attention_fusion_logic(uniform_modalities: List[np.ndarray]) -> None:
    """Test AttentionFusion computes weighted sum (mean in mock)."""
    fusion = AttentionFusion()
    fused = fusion.fuse(uniform_modalities)
    np.testing.assert_allclose(fused, np.ones((2, 4)) * 2.0)

@pytest.mark.parametrize("fusion_cls", [
    EarlyFusion, IntermediateFusion, LateFusion, AttentionFusion
])
def test_empty_modalities_error(fusion_cls: type) -> None:
    """Test all fusions raise error on empty list."""
    fusion = fusion_cls()
    with pytest.raises(ValueError, match="No modalities"):
        fusion.fuse([])

@pytest.mark.parametrize("fusion_cls", [
    IntermediateFusion, LateFusion, AttentionFusion
])
def test_fusion_commutativity(fusion_cls: type, uniform_modalities: List[np.ndarray]) -> None:
    """Test that symmetrical fusions are order-invariant."""
    fusion = fusion_cls()
    fused_1 = fusion.fuse(uniform_modalities)
    fused_2 = fusion.fuse(uniform_modalities[::-1])
    np.testing.assert_allclose(fused_1, fused_2)

@pytest.mark.parametrize("dim", [1, 2, 5, 10, 50, 100])
def test_early_fusion_large_dims(dim: int) -> None:
    """Test EarlyFusion handles large dimensions."""
    fusion = EarlyFusion()
    mods = [np.random.rand(1, dim), np.random.rand(1, dim)]
    res = fusion.fuse(mods)
    assert res.shape == (1, dim * 2)

@pytest.mark.parametrize("fusion_cls", [
    IntermediateFusion, LateFusion, AttentionFusion
])
def test_single_modality(fusion_cls: type) -> None:
    """Test fusions that allow single modality return identity."""
    fusion = fusion_cls()
    mod = [np.random.rand(4, 4)]
    res = fusion.fuse(mod)
    np.testing.assert_array_equal(res, mod[0])
