"""Self-supervised learning objectives."""

import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SSLObjective(abc.ABC):
    """Abstract base class for SSL objectives."""

    @abc.abstractmethod
    def compute_loss(self, view1_emb: np.ndarray, view2_emb: np.ndarray, **kwargs) -> float:
        """Compute the SSL loss."""
        pass

class ContrastiveLoss(SSLObjective):
    """InfoNCE / NT-Xent style loss."""

    def __init__(self, temperature: float = 0.07, epsilon: float = 1e-8):
        self.temperature = temperature
        self.epsilon = epsilon

    def compute_loss(self, view1_emb: np.ndarray, view2_emb: np.ndarray, **kwargs) -> float:
        # Normalize embeddings
        v1_norm = view1_emb / (np.linalg.norm(view1_emb, axis=1, keepdims=True) + self.epsilon)
        v2_norm = view2_emb / (np.linalg.norm(view2_emb, axis=1, keepdims=True) + self.epsilon)

        batch_size = view1_emb.shape[0]
        # Cosine similarity matrix
        sim_matrix = v1_norm @ v2_norm.T
        sim_matrix = sim_matrix / self.temperature

        # Positive pairs are on the diagonal
        positives = np.diag(sim_matrix)

        # LogSumExp trick for stability
        max_sim = np.max(sim_matrix, axis=1, keepdims=True)
        exp_sim = np.exp(sim_matrix - max_sim)
        sum_exp_sim = np.sum(exp_sim, axis=1)

        loss = -np.mean(positives - max_sim.squeeze() - np.log(sum_exp_sim + self.epsilon))
        return float(loss)

class PredictiveLoss(SSLObjective):
    """MSE between predicted and target features."""

    def compute_loss(self, view1_emb: np.ndarray, view2_emb: np.ndarray, **kwargs) -> float:
        diff = view1_emb - view2_emb
        loss = np.mean(diff ** 2)
        return float(loss)

class MaskedModelingLoss(SSLObjective):
    """MSE reconstruction loss on masked positions only."""

    def compute_loss(self, view1_emb: np.ndarray, view2_emb: np.ndarray, **kwargs) -> float:
        mask = kwargs.get('mask', None)
        if mask is None:
            mask = np.ones_like(view1_emb)
        diff = (view1_emb - view2_emb) * mask
        loss = np.sum(diff ** 2) / (np.sum(mask) + 1e-8)
        return float(loss)

class BootstrapLoss(SSLObjective):
    """Negative cosine similarity between online and target projections (BYOL-style)."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def compute_loss(self, view1_emb: np.ndarray, view2_emb: np.ndarray, **kwargs) -> float:
        v1_norm = view1_emb / (np.linalg.norm(view1_emb, axis=1, keepdims=True) + self.epsilon)
        v2_norm = view2_emb / (np.linalg.norm(view2_emb, axis=1, keepdims=True) + self.epsilon)
        sim = np.sum(v1_norm * v2_norm, axis=1)
        loss = -np.mean(sim)
        return float(loss)
