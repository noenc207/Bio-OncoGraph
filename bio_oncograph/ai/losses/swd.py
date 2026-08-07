"""Sliced Wasserstein Distance loss."""

import numpy as np
from bio_oncograph.ai.losses.base import LossFunction


class SWDLoss(LossFunction):
    """Sliced Wasserstein Distance loss function."""

    def __init__(self, num_projections: int = 50, p: int = 2) -> None:
        """Initialize SWD loss.

        Args:
            num_projections: Number of random projections to use.
            p: The power of the distance metric (usually 2).
        """
        super().__init__()
        self.num_projections = num_projections
        self.p = p

    def compute(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute the deterministic Sliced Wasserstein Distance.

        Args:
            outputs: Output distribution samples (shape: [N, D]).
            targets: Target distribution samples (shape: [M, D]).

        Returns:
            Scalar SWD value.
        """
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
            
        dim = outputs.shape[1]
        
        # Use a fixed random seed for deterministic projections
        rng = np.random.RandomState(42)
        
        # Generate normalized projection vectors
        projections = rng.randn(dim, self.num_projections)
        norms = np.linalg.norm(projections, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        projections /= norms
        
        # Project data
        # outputs shape: [N, D], projections shape: [D, P] -> [N, P]
        proj_outputs = np.dot(outputs, projections)
        proj_targets = np.dot(targets, projections)
        
        # Sort projections
        proj_outputs = np.sort(proj_outputs, axis=0)
        proj_targets = np.sort(proj_targets, axis=0)
        
        # Interpolate if sample sizes differ
        n_out = proj_outputs.shape[0]
        n_tar = proj_targets.shape[0]
        
        if n_out != n_tar:
            u_out = np.linspace(0, 1, n_out)
            u_tar = np.linspace(0, 1, n_tar)
            
            interpolated_targets = np.zeros_like(proj_outputs)
            for i in range(self.num_projections):
                interpolated_targets[:, i] = np.interp(u_out, u_tar, proj_targets[:, i])
            proj_targets = interpolated_targets
            
        # Compute distance
        distance = np.mean(np.abs(proj_outputs - proj_targets) ** self.p)
        return np.array(distance)
