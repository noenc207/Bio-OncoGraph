"""
Graph Construction Engine Configuration for Bio-OncoGraph.

Provides ``GraphConfig`` — an immutable dataclass containing
configuration for graph validation, construction, and serialization.

Design references:
  - Software Architecture § 22 (AI Modules — Graph)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    """
    Configuration for graph construction and validation.

    Attributes:
        directed: If ``True``, the graph is treated as directed.
            ``False`` implies edges (u,v) and (v,u) are equivalent
            or structurally enforced to be symmetric.
        allow_self_loop: If ``False``, validation will fail if
            an edge connects a node to itself.
        weighted: If ``True``, edges are expected to carry a
            weight attribute (used during validation/stats).
        connectivity_rule: The rule used when constructing edges
            (e.g., 'knn', 'radius', 'delaunay'). Used as metadata.
        serialization_version: Schema version for serialized graphs.
    """

    directed: bool = False
    allow_self_loop: bool = False
    weighted: bool = False
    connectivity_rule: str = "knn"
    serialization_version: str = "1.0.0"
