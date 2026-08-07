"""Module for validating constructed graphs."""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationReport:
    """Report containing the results of a graph validation.

    Attributes:
        is_valid: True if the graph passes all critical checks.
        errors: A tuple of error messages describing critical failures.
        warnings: A tuple of warning messages describing non-critical issues.
        statistics: A dictionary containing validation-related statistics.
    """

    is_valid: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    statistics: Dict[str, Any]


class GraphValidator:
    """Validator class for ensuring graph integrity."""

    def __init__(self) -> None:
        """Initialize the GraphValidator."""
        logger.info("Initializing GraphValidator.")

    def validate(self, graph: Any) -> ValidationReport:
        """Validate a graph's structure and properties.

        Args:
            graph: The graph object to validate. Expected to have 'num_nodes',
                   'edge_index', and 'metadata' attributes or keys.

        Returns:
            A ValidationReport detailing any issues found.
        """
        logger.info("Validating graph.")
        errors = []
        warnings = []
        stats = {}

        if isinstance(graph, dict):
            num_nodes = graph.get("num_nodes", 0)
            edge_index = graph.get("edge_index", np.empty((2, 0)))
            metadata = graph.get("metadata", {})
        else:
            num_nodes = getattr(graph, "num_nodes", 0)
            edge_index = getattr(graph, "edge_index", np.empty((2, 0)))
            metadata = getattr(graph, "metadata", {})

        stats["num_nodes"] = num_nodes
        stats["num_edges"] = edge_index.shape[1] if edge_index.ndim == 2 else 0

        if num_nodes == 0:
            errors.append("Graph is empty (0 nodes).")
            return ValidationReport(is_valid=False, errors=tuple(errors), warnings=tuple(warnings), statistics=stats)

        if edge_index.size > 0:
            min_idx = np.min(edge_index)
            max_idx = np.max(edge_index)
            if min_idx < 0 or max_idx >= num_nodes:
                errors.append(f"Edge indices out of range: min={min_idx}, max={max_idx}, allowed=[0, {num_nodes-1}].")

        if edge_index.size > 0:
            self_loops = int(np.sum(edge_index[0] == edge_index[1]))
            stats["self_loops"] = self_loops
            policy = metadata.get("self_loop_policy", "allowed")
            if self_loops > 0 and policy == "forbidden":
                errors.append(f"Graph contains {self_loops} self-loops but policy forbids them.")
            elif self_loops > 0:
                warnings.append(f"Graph contains {self_loops} self-loops.")

        if edge_index.size > 0:
            connected_nodes = np.unique(edge_index)
            isolated_count = num_nodes - len(connected_nodes)
            stats["isolated_nodes"] = isolated_count
            if isolated_count > 0:
                warnings.append(f"Graph contains {isolated_count} isolated nodes.")
        else:
            warnings.append("Graph has no edges (all nodes isolated).")

        is_valid = len(errors) == 0
        logger.info(f"Validation completed: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}.")
        
        return ValidationReport(
            is_valid=is_valid,
            errors=tuple(errors),
            warnings=tuple(warnings),
            statistics=stats,
        )
