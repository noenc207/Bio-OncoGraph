"""Scheduler factory module for creating learning rate schedulers."""

from typing import Any, Dict

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)


class SchedulerFactory:
    """Factory class for creating PyTorch learning rate schedulers."""

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        scheduler_params: Dict[str, Any],
        warmup_steps: int = 0,
        warmup_start_factor: float = 0.1,
    ) -> LRScheduler | ReduceLROnPlateau:
        """Creates a learning rate scheduler with optional linear warmup.

        Args:
            optimizer: The optimizer to attach the scheduler to.
            scheduler_type: The type of scheduler (e.g., 'cosine', 'plateau', 'linear').
            scheduler_params: Parameters for the chosen scheduler.
            warmup_steps: Number of warmup steps. If 0, no warmup is applied.
            warmup_start_factor: Starting multiplier for the learning rate during warmup.

        Returns:
            The created learning rate scheduler.

        Raises:
            ValueError: If an unsupported scheduler type is provided.
        """
        if scheduler_type.lower() == "cosine":
            main_scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_type.lower() == "plateau":
            if warmup_steps > 0:
                raise NotImplementedError("Warmup is not directly supported with ReduceLROnPlateau via SequentialLR.")
            return ReduceLROnPlateau(optimizer, **scheduler_params)
        elif scheduler_type.lower() == "linear":
            main_scheduler = LinearLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        
        return main_scheduler
