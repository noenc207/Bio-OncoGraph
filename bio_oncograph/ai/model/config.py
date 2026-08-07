"""Configuration classes for the Bio-OncoGraph M015 Model."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Bio-OncoGraph model architecture.
    
    Attributes:
        hidden_dim: The dimensionality of the hidden layers.
        num_layers: The number of layers in the graph neural network.
        pooling_type: The type of pooling to use ('mean', 'max', 'sum').
        dropout: The dropout probability.
        activation: The activation function to use ('relu', 'gelu').
        normalization: The normalization technique to use ('layer', 'batch', 'none').
        classification_head_enabled: Whether to enable the classification head.
        survival_head_enabled: Whether to enable the survival head.
        embedding_dim: The dimensionality of the node embeddings.
        init_method: The weight initialization method to use ('xavier', 'kaiming', 'orthogonal').
    """
    hidden_dim: int
    num_layers: int
    pooling_type: str
    dropout: float
    activation: str
    normalization: str
    classification_head_enabled: bool
    survival_head_enabled: bool
    embedding_dim: int
    init_method: str
