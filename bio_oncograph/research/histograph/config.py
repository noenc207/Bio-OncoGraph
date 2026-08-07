"""Configuration for histopathology graph construction."""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HistographConfig:
    """Configuration for histopathology graph construction.

    Attributes:
        graph_type: Type of the graph ('cell', 'tissue', 'region').
        edge_strategy: Strategy to construct edges ('knn', 'radius', 'delaunay').
        k_neighbors: Number of neighbors for KNN edge strategy.
        radius: Radius for radius-based edge strategy.
        self_loops: Whether to include self loops in the graph.
        directed: Whether the graph is directed.
        feature_dim: Dimensionality of node features.
        min_nodes: Minimum number of nodes required to build a valid graph.
        max_nodes: Maximum number of nodes allowed.
        seed: Random seed for reproducibility.
    """
    graph_type: str = 'cell'
    edge_strategy: str = 'knn'
    k_neighbors: int = 5
    radius: float = 50.0
    self_loops: bool = False
    directed: bool = False
    feature_dim: int = 64
    min_nodes: int = 1
    max_nodes: int = 100000
    seed: int = 42
