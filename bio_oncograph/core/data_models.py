"""
Data models defining the flow of tensors throughout the framework.
"""

from dataclasses import dataclass
from typing import List, Optional

from typing import Any


@dataclass
class SurvivalTarget:
    """Ground truth target for discrete-time survival prediction."""
    time_bins: Any
    event_indicators: Any


@dataclass
class BioGraphOutput:
    """Standardized output from the Bio-OncoGraph forward pass."""
    hazard_logits: Any
    survival_curves: Any
    persistence_diagrams: Optional[List[Any]] = None
    attention_weights: Optional[Any] = None


@dataclass(kw_only=True)
class BioGraphData:
    """
    Data object for holding Bio-OncoGraph specific tensors.
    """
    x: Optional[Any] = None
    pos: Optional[Any] = None
    edge_index: Optional[Any] = None
    survival_target: Optional[SurvivalTarget] = None
