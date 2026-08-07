"""Graph encoders for self-supervised pretraining."""

import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)

class GraphEncoder(abc.ABC):
    """Abstract base class for graph encoders."""

    @abc.abstractmethod
    def encode(self, node_features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """Encode node features using graph structure."""
        pass

    @abc.abstractmethod
    def reset_parameters(self) -> None:
        """Re-initialize parameters."""
        pass

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Output dimension of the encoder."""
        pass

class GCNEncoder(GraphEncoder):
    """Graph Convolutional Network encoder (Mock implementation)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, seed: int = 42):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._seed = seed
        self._weights = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng = np.random.default_rng(self._seed)
        self._weights = []
        dims = [self._input_dim] + [self._hidden_dim] * self._num_layers
        for i in range(self._num_layers):
            self._weights.append(rng.normal(0, 0.1, (dims[i], dims[i+1])))

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    def encode(self, node_features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        x = node_features
        num_nodes = x.shape[0]
        for w in self._weights:
            x = x @ w
            # Mock mean aggregation
            agg = np.zeros_like(x)
            counts = np.zeros(num_nodes)
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                agg[dst] += x[src]
                counts[dst] += 1
            counts[counts == 0] = 1
            agg = agg / counts[:, None]
            x = np.maximum(0, agg)  # ReLU
        return x

class GATEncoder(GraphEncoder):
    """Graph Attention Network encoder (Mock implementation)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, seed: int = 42):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._seed = seed
        self._weights = []
        self._att_weights = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng = np.random.default_rng(self._seed)
        self._weights = []
        self._att_weights = []
        dims = [self._input_dim] + [self._hidden_dim] * self._num_layers
        for i in range(self._num_layers):
            self._weights.append(rng.normal(0, 0.1, (dims[i], dims[i+1])))
            self._att_weights.append(rng.normal(0, 0.1, (2 * dims[i+1], 1)))

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    def encode(self, node_features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        x = node_features
        num_nodes = x.shape[0]
        for w, a in zip(self._weights, self._att_weights):
            x = x @ w
            agg = np.zeros_like(x)
            # Simplified attention over neighbors (just average for mock)
            counts = np.zeros(num_nodes)
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                agg[dst] += x[src]
                counts[dst] += 1
            counts[counts == 0] = 1
            agg = agg / counts[:, None]
            x = np.maximum(0, agg)
        return x

class TransformerEncoder(GraphEncoder):
    """Transformer-based Graph encoder (Mock implementation)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, seed: int = 42):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._seed = seed
        self._weights = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng = np.random.default_rng(self._seed)
        self._weights = []
        dims = [self._input_dim] + [self._hidden_dim] * self._num_layers
        for i in range(self._num_layers):
            self._weights.append(rng.normal(0, 0.1, (dims[i], dims[i+1])))

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    def encode(self, node_features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        x = node_features
        for w in self._weights:
            x = x @ w
            # Mock self attention over all nodes (global mean pool + broadcast)
            global_mean = np.mean(x, axis=0)
            x = x + global_mean  # Residual-like addition
            x = np.maximum(0, x)
        return x

class MPNNEncoder(GraphEncoder):
    """Message Passing Neural Network encoder (Mock implementation)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, seed: int = 42):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._seed = seed
        self._weights = []
        self.reset_parameters()

    def reset_parameters(self) -> None:
        rng = np.random.default_rng(self._seed)
        self._weights = []
        dims = [self._input_dim] + [self._hidden_dim] * self._num_layers
        for i in range(self._num_layers):
            self._weights.append(rng.normal(0, 0.1, (dims[i], dims[i+1])))

    @property
    def output_dim(self) -> int:
        return self._hidden_dim

    def encode(self, node_features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        x = node_features
        for w in self._weights:
            x = x @ w
            agg = np.zeros_like(x)
            for j in range(edge_index.shape[1]):
                src, dst = edge_index[0, j], edge_index[1, j]
                agg[dst] += x[src]
            x = x + agg
            x = np.maximum(0, x)
        return x
