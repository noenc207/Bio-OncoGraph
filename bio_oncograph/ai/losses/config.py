"""Loss configuration."""

import dataclasses

@dataclasses.dataclass(frozen=True)
class LossConfig:
    """Configuration for loss functions.
    
    Attributes:
        epsilon: Small constant for numerical stability.
        reduction: Method to reduce the loss ('mean', 'sum', 'none').
        enable_swd: Whether to enable Sliced Wasserstein Distance.
        enable_survival: Whether to enable survival loss components.
        regularization_weight: Weight for regularization terms.
    """
    epsilon: float = 1e-8
    reduction: str = 'mean'
    enable_swd: bool = False
    enable_survival: bool = False
    regularization_weight: float = 1e-4
