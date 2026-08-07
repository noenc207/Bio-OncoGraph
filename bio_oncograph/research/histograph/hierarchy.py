"""Module for hierarchical graph representations."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HierarchicalGraph:
    """Represents a multi-level hierarchical graph.

    Attributes:
        cell_graph: The graph at the cell level.
        tissue_graph: The graph at the tissue level.
        region_graph: The graph at the region level.
        cross_level_edges: A numpy array of shape [2, E] mapping child -> parent node indices.
        metadata: A dictionary containing additional metadata for the hierarchy.
    """

    cell_graph: Optional[Any] = None
    tissue_graph: Optional[Any] = None
    region_graph: Optional[Any] = None
    cross_level_edges: np.ndarray = field(default_factory=lambda: np.empty((2, 0), dtype=np.int64))
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchyBuilder:
    """Builder class for constructing hierarchical graphs."""

    def __init__(self) -> None:
        """Initialize the HierarchyBuilder."""
        logger.info("Initializing HierarchyBuilder.")

    def build(
        self,
        cell_graph: Optional[Any],
        tissue_graph: Optional[Any],
        region_graph: Optional[Any],
        assignment_strategy: str = "nearest"
    ) -> HierarchicalGraph:
        """Build a hierarchical graph from individual level graphs.

        Args:
            cell_graph: Optional cell-level graph.
            tissue_graph: Optional tissue-level graph.
            region_graph: Optional region-level graph.
            assignment_strategy: The strategy to use for assigning nodes across levels.

        Returns:
            A HierarchicalGraph instance.
        """
        logger.info(f"Building hierarchical graph with strategy '{assignment_strategy}'.")
        cross_level_edges = np.empty((2, 0), dtype=np.int64)

        return HierarchicalGraph(
            cell_graph=cell_graph,
            tissue_graph=tissue_graph,
            region_graph=region_graph,
            cross_level_edges=cross_level_edges,
            metadata={"strategy": assignment_strategy},
        )

    def assign_cells_to_tissue(self, cell_centroids: np.ndarray, tissue_centroids: np.ndarray) -> np.ndarray:
        """Assign cells to their nearest tissue component.

        Args:
            cell_centroids: A numpy array of shape (N_c, D) with cell coordinates.
            tissue_centroids: A numpy array of shape (N_t, D) with tissue coordinates.

        Returns:
            A numpy array of shape (N_c,) containing the index of the assigned tissue for each cell.
        """
        logger.debug(f"Assigning {cell_centroids.shape[0]} cells to {tissue_centroids.shape[0]} tissues.")
        if tissue_centroids.shape[0] == 0 or cell_centroids.shape[0] == 0:
            return np.array([], dtype=np.int64)

        diff = cell_centroids[:, np.newaxis, :] - tissue_centroids[np.newaxis, :, :]
        distances = np.sum(diff**2, axis=-1)
        assignments = np.argmin(distances, axis=-1)
        return assignments

    def assign_tissue_to_region(self, tissue_centroids: np.ndarray, region_bboxes: np.ndarray) -> np.ndarray:
        """Assign tissue components to regions based on containment.

        Args:
            tissue_centroids: A numpy array of shape (N_t, D) with tissue coordinates.
            region_bboxes: A numpy array of shape (N_r, 2*D) representing bounding boxes [min_x, min_y, max_x, max_y].

        Returns:
            A numpy array of shape (N_t,) containing the assigned region index, or -1 if unassigned.
        """
        logger.debug(f"Assigning {tissue_centroids.shape[0]} tissues to {region_bboxes.shape[0]} regions.")
        num_tissues = tissue_centroids.shape[0]
        num_regions = region_bboxes.shape[0]
        assignments = np.full(num_tissues, -1, dtype=np.int64)

        if num_tissues == 0 or num_regions == 0:
            return assignments

        for i in range(num_tissues):
            x, y = tissue_centroids[i]
            for j in range(num_regions):
                min_x, min_y, max_x, max_y = region_bboxes[j]
                if min_x <= x <= max_x and min_y <= y <= max_y:
                    assignments[i] = j
                    break

        return assignments
