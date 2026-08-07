"""PyTorch Automatic Mixed Precision (AMP) utilities."""
import torch
from typing import Any, Callable, TypeVar, Generator
from contextlib import contextmanager

T = TypeVar('T')


class GradScalerWrapper:
    """Wrapper for PyTorch GradScaler to manage mixed precision gradients."""

    def __init__(self, enabled: bool = True) -> None:
        """Initializes the GradScalerWrapper.

        Args:
            enabled (bool): Whether to enable the GradScaler.
        """
        self.scaler = torch.amp.GradScaler('cuda', enabled=enabled)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss tensor for backward pass.

        Args:
            loss (torch.Tensor): The loss tensor to scale.

        Returns:
            torch.Tensor: The scaled loss tensor.
        """
        return self.scaler.scale(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Steps the optimizer using scaled gradients.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to step.
        """
        self.scaler.step(optimizer)

    def update(self) -> None:
        """Updates the scale factor for the next iteration."""
        self.scaler.update()


@contextmanager
def autocast(device_type: str = 'cuda', enabled: bool = True, dtype: torch.dtype = torch.float16) -> Generator[None, None, None]:
    """Context manager for PyTorch automatic mixed precision.

    Args:
        device_type (str): The device type to use for autocast ('cuda' or 'cpu').
        enabled (bool): Whether autocasting should be enabled in the region.
        dtype (torch.dtype): The data type to cast to.

    Yields:
        None
    """
    with torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype):
        yield
