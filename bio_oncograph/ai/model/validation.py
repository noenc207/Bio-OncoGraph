"""Validation utilities for the Bio-OncoGraph M015 Model inputs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bio_oncograph.ai.graph.graph import Graph
    from bio_oncograph.ai.model.config import ModelConfig


def validate_model_input(graph: 'Graph', config: 'ModelConfig') -> None:
    """Validates the input graph against the model configuration.
    
    Args:
        graph: The input graph to validate.
        config: The model configuration.
        
    Raises:
        ValueError: If the graph is invalid or incompatible with the config.
    """
    if not graph.nodes:
        raise ValueError("Graph must not be empty. It contains no nodes.")
        
    # Check node features and expected dimensions
    expected_dim = config.embedding_dim
    for node_id, node_data in graph.nodes.items():
        if 'features' not in node_data:
            raise ValueError(f"Node {node_id} is missing 'features' attribute.")
        
        features = node_data['features']
        try:
            feature_shape = features.shape
        except AttributeError:
            raise ValueError(f"Features for node {node_id} must be a numpy array-like object.")
            
        if len(feature_shape) != 1 or feature_shape[0] != expected_dim:
            raise ValueError(
                f"Node {node_id} features have shape {feature_shape}, "
                f"expected ({expected_dim},)."
            )
            
    # Verify edges are valid (source and target exist in nodes)
    for u, v in graph.edges:
        if u not in graph.nodes:
            raise ValueError(f"Edge ({u}, {v}) contains invalid source node {u}.")
        if v not in graph.nodes:
            raise ValueError(f"Edge ({u}, {v}) contains invalid target node {v}.")
