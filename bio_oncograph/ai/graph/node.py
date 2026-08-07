"""
Node data structure for Bio-OncoGraph graphs.

Provides ``Node`` — an immutable representation of a graph node.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Node:
    """
    Immutable graph node.

    Attributes:
        id: Unique identifier for the node (usually an integer
            or string).
        features: Dictionary of node attributes/features (e.g.,
            coordinates, morphological features, embeddings).
    """

    id: Any
    features: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id
