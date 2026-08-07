"""Graph augmentations for self-supervised pretraining."""

import abc
import logging
from typing import Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

class GraphAugmentation(abc.ABC):
    """Abstract base class for graph augmentations."""

    @abc.abstractmethod
    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to node features and edge index."""
        pass

class NodeDropout(GraphAugmentation):
    """Randomly zero-out node features."""

    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        mask = rng.random(node_features.shape[0]) > self.dropout_rate
        new_features = node_features.copy()
        new_features[~mask] = 0.0
        return new_features, edge_index

class EdgeDropout(GraphAugmentation):
    """Randomly remove edges."""

    def __init__(self, dropout_rate: float = 0.1):
        self.dropout_rate = dropout_rate

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        num_edges = edge_index.shape[1]
        mask = rng.random(num_edges) > self.dropout_rate
        new_edge_index = edge_index[:, mask]
        return node_features, new_edge_index

class FeatureMasking(GraphAugmentation):
    """Mask random feature dimensions to zero."""

    def __init__(self, mask_rate: float = 0.15):
        self.mask_rate = mask_rate

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        mask = rng.random(node_features.shape) > self.mask_rate
        new_features = node_features * mask
        return new_features, edge_index

class SubgraphSampling(GraphAugmentation):
    """Sample a connected subgraph (mock random subset)."""

    def __init__(self, fraction: float = 0.8):
        self.fraction = fraction

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        num_nodes = node_features.shape[0]
        keep_nodes = rng.choice(num_nodes, size=int(num_nodes * self.fraction), replace=False)
        keep_set = set(keep_nodes)
        
        # Filter edges
        mask = np.array([src in keep_set and dst in keep_set for src, dst in zip(edge_index[0], edge_index[1])])
        new_edge_index = edge_index[:, mask]
        
        return node_features.copy(), new_edge_index

class AttributePerturbation(GraphAugmentation):
    """Add Gaussian noise to features."""

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self.noise_std, node_features.shape)
        new_features = node_features + noise
        return new_features, edge_index

class AugmentationPipeline:
    """Compose multiple augmentations sequentially."""

    def __init__(self, augmentations: List[GraphAugmentation]):
        self.augmentations = augmentations

    def augment(self, node_features: np.ndarray, edge_index: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        features, edges = node_features, edge_index
        for i, aug in enumerate(self.augmentations):
            features, edges = aug.augment(features, edges, seed + i)
        return features, edges
