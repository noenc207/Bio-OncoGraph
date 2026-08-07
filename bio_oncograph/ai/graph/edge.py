"""
Edge data structure for Bio-OncoGraph graphs.

Provides ``Edge`` — an immutable representation of a graph edge.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Edge:
    """
    Immutable graph edge.

    Attributes:
        id: Unique identifier for the edge.
        source_id: Identifier of the source node.
        target_id: Identifier of the target node.
        weight: Edge weight (defaults to 1.0 for unweighted graphs).
        features: Dictionary of edge attributes (e.g., distance,
            type, interaction scores).
    """

    id: Any
    source_id: Any
    target_id: Any
    weight: float = 1.0
    features: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return (
            self.id == other.id
            and self.source_id == other.source_id
            and self.target_id == other.target_id
        )
