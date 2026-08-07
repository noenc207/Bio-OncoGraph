"""PyTorch backend facade for Bio-OncoGraph."""
import torch
from typing import Any, Dict, List, Optional
from bio_oncograph.infrastructure.pytorch.device import get_device
from bio_oncograph.infrastructure.pytorch.initialization import seed_everything
from bio_oncograph.infrastructure.pytorch.amp import GradScalerWrapper, autocast
from bio_oncograph.infrastructure.pytorch.tensors import create_tensor, features_to_tensor, edges_to_tensor


class PyTorchBackend:
    """Main facade class orchestrating PyTorch devices, tensors, and AMP."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initializes the PyTorch backend.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.device = get_device()
        self.scaler = GradScalerWrapper(enabled=self.device.type == 'cuda')
        
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Sets the random seed for reproducibility.

        Args:
            seed (int): The random seed.
        """
        seed_everything(seed)

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves a tensor to the configured device.

        Args:
            tensor (torch.Tensor): The tensor to move.

        Returns:
            torch.Tensor: The tensor on the configured device.
        """
        return tensor.to(self.device)

    def create_features(self, features: List[List[float]]) -> torch.Tensor:
        """Creates a feature tensor on the backend's device.

        Args:
            features (List[List[float]]): The input features.

        Returns:
            torch.Tensor: The feature tensor.
        """
        return features_to_tensor(features, device=self.device)

    def create_edges(self, edges: List[tuple[int, int]]) -> torch.Tensor:
        """Creates an edge index tensor on the backend's device.

        Args:
            edges (List[tuple[int, int]]): The input edges.

        Returns:
            torch.Tensor: The edge index tensor.
        """
        return edges_to_tensor(edges, device=self.device)

    def get_autocast(self, enabled: bool = True) -> Any:
        """Gets the autocast context manager.

        Args:
            enabled (bool): Whether to enable mixed precision.

        Returns:
            Any: The autocast context manager.
        """
        return autocast(device_type=self.device.type, enabled=enabled)
