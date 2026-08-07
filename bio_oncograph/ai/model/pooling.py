"""Graph Pooling module for Bio-OncoGraph."""
import numpy as np
from typing import Dict, Any


class GraphPooling:
    """Graph Pooling layer for aggregating node embeddings into graph embeddings.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        pooling_type (str): Type of pooling ('mean', 'max', 'sum').
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize GraphPooling layer.

        Args:
            config: Configuration dictionary containing 'pooling_type'.
        """
        self.config = config
        self.pooling_type = config.get("pooling_type", "mean")
        if self.pooling_type not in ["mean", "max", "sum"]:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    def forward(self, node_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate node embeddings into a single graph representation.

        Args:
            node_embeddings: Dictionary mapping node IDs to their embeddings.

        Returns:
            A 1D numpy array representing the pooled graph embedding.
        """
        if not node_embeddings:
            raise ValueError("Empty node embeddings provided for pooling.")
            
        embeddings_list = list(node_embeddings.values())
        embeddings_array = np.stack(embeddings_list, axis=0)
        
        if self.pooling_type == "mean":
            return np.mean(embeddings_array, axis=0)
        elif self.pooling_type == "max":
            return np.max(embeddings_array, axis=0)
        else:  # sum
            return np.sum(embeddings_array, axis=0)
