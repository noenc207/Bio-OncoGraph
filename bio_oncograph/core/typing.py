"""
Type definitions, TypedDicts, and NamedTuples for Bio-OncoGraph.
"""

from typing import Any


class MetricsDict(TypedDict):
    """Dictionary structure for evaluation metrics."""
    c_index: float
    brier_score: float
    d_calibration: float


class LossOutput(NamedTuple):
    """Output from the CalTopoSurv loss module."""
    total_loss: Any
    nll: Any
    ranking: Any
    topo: Any


class PoolingLog(NamedTuple):
    """Logging payload emitted by topological pooling layers."""
    retained_nodes: int
    betti_0_error: float
