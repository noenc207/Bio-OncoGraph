"""Checkpoint management."""
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, save_dir: str) -> None:
        """Initializes the CheckpointManager.

        Args:
            save_dir: Directory to save the checkpoints.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save(
        self,
        filename: str,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves a checkpoint to disk.

        Args:
            filename: Name of the file to save.
            epoch: Current epoch number.
            model: The neural network model.
            optimizer: The optimizer.
            scheduler: The learning rate scheduler.
            scaler: Gradient scaler for mixed precision.
            extras: Any extra information to save (e.g., best metric).
        """
        filepath = os.path.join(self.save_dir, filename)
        
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        if extras is not None:
            state["extras"] = extras

        torch.save(state, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, Any]:
        """Loads a checkpoint from disk.

        Args:
            filepath: Path to the checkpoint file.
            model: The model to load weights into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.
            scaler: Scaler to load state into.

        Returns:
            A dictionary containing the epoch and any extras.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        logger.info(f"Loading checkpoint from {filepath}")
        state = torch.load(filepath, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
            
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])

        if "random_state" in state:
            random.setstate(state["random_state"])
        if "np_random_state" in state:
            np.random.set_state(state["np_random_state"])
        if "torch_random_state" in state:
            torch.set_rng_state(state["torch_random_state"])
        if "torch_cuda_random_state" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_random_state"])

        return {
            "epoch": state.get("epoch", 0),
            "extras": state.get("extras", {}),
        }
