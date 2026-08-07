"""
Tensor validation utilities for Bio-OncoGraph.

These guards implement the Fail-Fast engineering constraint from the
Design Freeze.  Every assertion is designed to fire at the *input*
boundary of a module's ``forward`` method, catching shape mismatches
and degenerate values before they propagate silently through the
computation graph.

A global flag ``VALIDATION_ENABLED`` controls whether checks run.
It should be set to ``False`` only during optimised inference to
eliminate overhead.
"""

import torch
from torch import Tensor

from bio_oncograph.core.exceptions import (
    ContractViolationError,
    NaNGradientException,
)

VALIDATION_ENABLED: bool = True


def validate_node_features(x: Tensor) -> None:
    """
    Validate that a node feature matrix has the correct shape and values.

    Checks:
      - Tensor is 2-dimensional ``(N, D)``.
      - Contains no NaN values.
      - Contains no Inf values.

    Args:
        x: Node feature tensor to validate.

    Raises:
        ContractViolationError: If the tensor is not 2D.
        NaNGradientException: If the tensor contains NaN or Inf.
    """
    if not VALIDATION_ENABLED:
        return

    if x.ndim != 2:
        raise ContractViolationError(
            f"Node features must be 2D (N, D), got shape {tuple(x.shape)}."
        )
    if torch.isnan(x).any():
        raise NaNGradientException("Node features contain NaN values.")
    if torch.isinf(x).any():
        raise NaNGradientException("Node features contain Inf values.")


def validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    """
    Validate that an edge index tensor is structurally correct.

    Checks:
      - Tensor is 2-dimensional with first dimension equal to 2.
      - All indices are in the range ``[0, num_nodes)``.

    Args:
        edge_index: Edge connectivity tensor ``(2, E)``.
        num_nodes: The total number of nodes in the graph.

    Raises:
        ContractViolationError: If shape or value constraints are
            violated.
    """
    if not VALIDATION_ENABLED:
        return

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ContractViolationError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}."
        )
    if edge_index.numel() > 0 and edge_index.max().item() >= num_nodes:
        raise ContractViolationError(
            f"edge_index contains index {edge_index.max().item()} "
            f"but num_nodes is {num_nodes}."
        )
    if edge_index.numel() > 0 and edge_index.min().item() < 0:
        raise ContractViolationError(
            "edge_index contains negative indices."
        )


def validate_batch_vector(batch: Tensor, num_nodes: int) -> None:
    """
    Validate that a batch assignment vector is correctly formed.

    Checks:
      - Tensor is 1-dimensional with length equal to ``num_nodes``.
      - Values are non-negative.
      - Values are non-decreasing (required by PyG batching convention).

    Args:
        batch: Batch assignment vector ``(N,)``.
        num_nodes: Expected number of nodes.

    Raises:
        ContractViolationError: If any constraint is violated.
    """
    if not VALIDATION_ENABLED:
        return

    if batch.ndim != 1:
        raise ContractViolationError(
            f"Batch vector must be 1D, got {batch.ndim}D."
        )
    if batch.size(0) != num_nodes:
        raise ContractViolationError(
            f"Batch vector length {batch.size(0)} != num_nodes {num_nodes}."
        )
    if batch.numel() > 0 and batch.min().item() < 0:
        raise ContractViolationError("Batch vector contains negative values.")
    if batch.numel() > 1:
        diffs = batch[1:] - batch[:-1]
        if (diffs < 0).any():
            raise ContractViolationError(
                "Batch vector is not non-decreasing."
            )


def validate_hazard_logits(logits: Tensor, bound: float = 1e4) -> None:
    """
    Validate that hazard logits have not diverged.

    Args:
        logits: Hazard logit tensor ``(B, T)``.
        bound: Absolute bound beyond which logits are considered diverged.

    Raises:
        ContractViolationError: If logits are not 2D.
        NaNGradientException: If logits contain NaN or exceed the bound.
    """
    if not VALIDATION_ENABLED:
        return

    if logits.ndim != 2:
        raise ContractViolationError(
            f"Hazard logits must be 2D (B, T), got shape {tuple(logits.shape)}."
        )
    if torch.isnan(logits).any():
        raise NaNGradientException("Hazard logits contain NaN.")
    if torch.any(logits.abs() > bound):
        raise NaNGradientException(
            f"Hazard logits diverged beyond ±{bound}."
        )
