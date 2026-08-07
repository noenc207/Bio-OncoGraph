"""Automatic Mixed Precision (AMP) management module."""

from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast


class AMPManager:
    """Manages Automatic Mixed Precision (AMP), gradient scaling, and accumulation."""

    def __init__(self, enabled: bool = True, grad_clip_val: Optional[float] = None) -> None:
        """Initializes the AMPManager.

        Args:
            enabled: Whether AMP is enabled.
            grad_clip_val: Maximum norm for gradient clipping. If None, no clipping is applied.
        """
        self.enabled = enabled
        self.grad_clip_val = grad_clip_val
        self.scaler = GradScaler(enabled=self.enabled)

    def autocast_context(self) -> autocast:
        """Returns the autocast context manager.

        Returns:
            The autocast context manager.
        """
        return autocast(enabled=self.enabled)

    def backward(self, loss: torch.Tensor) -> None:
        """Performs scaled backward pass.

        Args:
            loss: The loss tensor to call backward on.
        """
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer, model: Optional[torch.nn.Module] = None) -> None:
        """Unscales gradients, clips them, and performs an optimizer step.

        Args:
            optimizer: The optimizer to step.
            model: The model containing parameters (required if gradient clipping is enabled).
        """
        if self.grad_clip_val is not None and model is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_val)
            
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
