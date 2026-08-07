"""Feature Fusion module for Bio-OncoGraph."""
import numpy as np
from typing import Dict, Any


class FeatureFusion:
    """Layer to fuse graph embeddings with auxiliary/clinical features.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        weights (np.ndarray): Weight matrix for the linear projection.
        bias (np.ndarray): Bias vector for the linear projection.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize FeatureFusion layer.

        Args:
            config: Configuration dictionary containing 'graph_dim', 'aux_dim', 'output_dim', and 'seed'.
        """
        self.config = config
        seed = config.get("seed", 42)
        np.random.seed(seed)
        
        graph_dim = config.get("graph_dim", 64)
        aux_dim = config.get("aux_dim", 32)
        output_dim = config.get("output_dim", 64)
        
        input_dim = graph_dim + aux_dim
        
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / (input_dim + output_dim))
        self.bias = np.zeros(output_dim)

    def forward(self, graph_embedding: np.ndarray, auxiliary_features: np.ndarray) -> np.ndarray:
        """Fuse the graph embedding with auxiliary features.

        Args:
            graph_embedding: A 1D numpy array representing the graph.
            auxiliary_features: A 1D numpy array representing clinical/auxiliary data.

        Returns:
            A 1D numpy array of the fused features.
        """
        concatenated = np.concatenate([graph_embedding, auxiliary_features], axis=-1)
        fused = concatenated.dot(self.weights) + self.bias
        # Optional activation (e.g., ReLU)
        return np.maximum(0, fused)
