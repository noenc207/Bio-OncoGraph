"""Graph Encoder module for Bio-OncoGraph."""
import numpy as np
from typing import Dict, Any, List, Tuple


class GraphEncoder:
    """Multi-layer Graph Convolutional Network encoder using numpy.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        weights (List[np.ndarray]): List of weight matrices for each layer.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the GraphEncoder.

        Args:
            config: Dictionary containing 'input_dim', 'hidden_dims', and 'seed'.
        """
        self.config = config
        seed = config.get("seed", 42)
        np.random.seed(seed)
        
        input_dim = config.get("input_dim", 64)
        hidden_dims = config.get("hidden_dims", [64, 64])
        
        self.weights: List[np.ndarray] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            # Initialize weights with standard normal distribution
            w = np.random.randn(prev_dim, dim) * np.sqrt(2.0 / (prev_dim + dim))
            self.weights.append(w)
            prev_dim = dim

    def forward(self, graph: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Perform forward pass of the GCN.

        Args:
            graph: A dictionary with 'nodes' mapping node IDs to feature arrays,
                   and 'edges' containing a list of (source, target) tuples.

        Returns:
            A dictionary mapping node IDs to their computed embeddings.
        """
        nodes: Dict[str, np.ndarray] = graph.get("nodes", {})
        edges: List[Tuple[str, str]] = graph.get("edges", [])
        
        node_ids = list(nodes.keys())
        if not node_ids:
            return {}
            
        num_nodes = len(node_ids)
        node_idx_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Build adjacency matrix
        adj = np.zeros((num_nodes, num_nodes))
        for u, v in edges:
            if u in node_idx_map and v in node_idx_map:
                adj[node_idx_map[u], node_idx_map[v]] = 1.0
                adj[node_idx_map[v], node_idx_map[u]] = 1.0  # Assume undirected
                
        # Add self-loops
        adj += np.eye(num_nodes)
        
        # Degree matrix for normalization
        degree = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Normalized adjacency: D^{-1/2} A D^{-1/2}
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Feature matrix
        feature_dim = nodes[node_ids[0]].shape[0]
        h = np.zeros((num_nodes, feature_dim))
        for node_id, feat in nodes.items():
            h[node_idx_map[node_id]] = feat
            
        # GCN layers
        for w in self.weights:
            h = norm_adj.dot(h).dot(w)
            # ReLU activation
            h = np.maximum(0, h)
            
        # Format output
        embeddings = {}
        for i, node_id in enumerate(node_ids):
            embeddings[node_id] = h[i]
            
        return embeddings
