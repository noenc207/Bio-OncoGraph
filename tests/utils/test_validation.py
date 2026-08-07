"""
Unit tests for bio_oncograph.utils.validation.

Covers normal inputs, edge cases (empty tensors, single-element),
and all failure modes that must raise the correct exception type.
"""

import pytest
import torch

from bio_oncograph.core.exceptions import (
    ContractViolationError,
    NaNGradientException,
)
from bio_oncograph.utils import validation
from bio_oncograph.utils.validation import (
    validate_batch_vector,
    validate_edge_index,
    validate_hazard_logits,
    validate_node_features,
)


@pytest.fixture(autouse=True)
def enable_validation() -> None:
    """Ensure validation is enabled for every test."""
    validation.VALIDATION_ENABLED = True


# ── validate_node_features ──────────────────────────────────────────


def test_node_features_valid() -> None:
    """Valid 2D tensor passes without exception."""
    validate_node_features(torch.randn(10, 64))


def test_node_features_wrong_ndim() -> None:
    """3D tensor raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="2D"):
        validate_node_features(torch.randn(2, 10, 64))


def test_node_features_nan() -> None:
    """Tensor with NaN raises NaNGradientException."""
    x = torch.randn(5, 16)
    x[2, 3] = float("nan")
    with pytest.raises(NaNGradientException, match="NaN"):
        validate_node_features(x)


def test_node_features_inf() -> None:
    """Tensor with Inf raises NaNGradientException."""
    x = torch.randn(5, 16)
    x[0, 0] = float("inf")
    with pytest.raises(NaNGradientException, match="Inf"):
        validate_node_features(x)


def test_node_features_disabled() -> None:
    """No exception when VALIDATION_ENABLED is False."""
    validation.VALIDATION_ENABLED = False
    validate_node_features(torch.randn(2, 3, 4))  # wrong shape, should pass


# ── validate_edge_index ─────────────────────────────────────────────


def test_edge_index_valid() -> None:
    """Valid edge index passes."""
    validate_edge_index(torch.tensor([[0, 1], [1, 2]]).T, num_nodes=3)


def test_edge_index_wrong_shape() -> None:
    """Non-(2, E) shape raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="shape \\(2, E\\)"):
        validate_edge_index(torch.randint(0, 5, (3, 10)), num_nodes=5)


def test_edge_index_out_of_bounds() -> None:
    """Index >= num_nodes raises ContractViolationError."""
    edge_index = torch.tensor([[0, 5], [1, 2]])
    with pytest.raises(ContractViolationError, match="index 5"):
        validate_edge_index(edge_index, num_nodes=5)


def test_edge_index_negative() -> None:
    """Negative index raises ContractViolationError."""
    edge_index = torch.tensor([[0, -1], [1, 2]])
    with pytest.raises(ContractViolationError, match="negative"):
        validate_edge_index(edge_index, num_nodes=5)


def test_edge_index_empty() -> None:
    """Empty edge index (0 edges) passes validation."""
    validate_edge_index(torch.zeros(2, 0, dtype=torch.long), num_nodes=5)


# ── validate_batch_vector ───────────────────────────────────────────


def test_batch_vector_valid() -> None:
    """Correctly formed batch vector passes."""
    batch = torch.tensor([0, 0, 0, 1, 1, 2])
    validate_batch_vector(batch, num_nodes=6)


def test_batch_vector_wrong_length() -> None:
    """Length mismatch raises ContractViolationError."""
    batch = torch.tensor([0, 0, 1])
    with pytest.raises(ContractViolationError, match="length"):
        validate_batch_vector(batch, num_nodes=5)


def test_batch_vector_negative() -> None:
    """Negative value raises ContractViolationError."""
    batch = torch.tensor([-1, 0, 1])
    with pytest.raises(ContractViolationError, match="negative"):
        validate_batch_vector(batch, num_nodes=3)


def test_batch_vector_not_nondecreasing() -> None:
    """Non-monotonic vector raises ContractViolationError."""
    batch = torch.tensor([0, 2, 1])
    with pytest.raises(ContractViolationError, match="non-decreasing"):
        validate_batch_vector(batch, num_nodes=3)


def test_batch_vector_wrong_ndim() -> None:
    """2D tensor raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="1D"):
        validate_batch_vector(torch.zeros(3, 1, dtype=torch.long), num_nodes=3)


# ── validate_hazard_logits ──────────────────────────────────────────


def test_hazard_logits_valid() -> None:
    """Normal logits pass."""
    validate_hazard_logits(torch.randn(4, 10))


def test_hazard_logits_wrong_ndim() -> None:
    """1D tensor raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="2D"):
        validate_hazard_logits(torch.randn(10))


def test_hazard_logits_nan() -> None:
    """NaN in logits raises NaNGradientException."""
    logits = torch.randn(2, 5)
    logits[0, 0] = float("nan")
    with pytest.raises(NaNGradientException, match="NaN"):
        validate_hazard_logits(logits)


def test_hazard_logits_diverged() -> None:
    """Logits exceeding bound raises NaNGradientException."""
    logits = torch.zeros(2, 5)
    logits[1, 3] = 2e4
    with pytest.raises(NaNGradientException, match="diverged"):
        validate_hazard_logits(logits, bound=1e4)
