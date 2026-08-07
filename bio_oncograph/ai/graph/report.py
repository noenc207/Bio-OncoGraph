"""
Graph construction report for Bio-OncoGraph.

Provides ``GraphReport`` — an immutable container holding the graph,
its validation status, statistics, and metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from bio_oncograph.ai.graph.graph import Graph
from bio_oncograph.ai.graph.statistics import GraphStatistics


@dataclass(frozen=True)
class GraphReport:
    """
    Immutable result of graph construction.

    Attributes:
        graph: The constructed ``Graph`` object.
        is_valid: ``True`` if the graph passed all validation checks.
        statistics: Computed ``GraphStatistics``.
        warnings: Tuple of warning messages during construction or
            validation.
        metadata: Extensible metadata dictionary.
    """

    graph: Graph
    is_valid: bool
    statistics: GraphStatistics
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.is_valid, self.warnings))
