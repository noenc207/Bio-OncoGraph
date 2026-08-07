"""Checkpoint saving and loading utilities for Bio-OncoGraph PyTorch backend."""

import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Saves a training checkpoint.
    
    Args:
        path: File path to save the checkpoint.
        model: The model to save. Will save its state_dict.
        optimizer: Optional optimizer to save its state_dict.
        scheduler: Optional learning rate scheduler to save its state_dict.
        epoch: Optional current epoch number.
        step: Optional current training step number.
        metadata: Optional additional metadata dictionary to save.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if step is not None:
        checkpoint["step"] = step
    if metadata is not None:
        checkpoint["metadata"] = metadata
        
    torch.save(checkpoint, path)

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Loads a training checkpoint.
    
    Args:
        path: File path to load the checkpoint from.
        model: The model to load the state_dict into.
        optimizer: Optional optimizer to load the state_dict into.
        scheduler: Optional scheduler to load the state_dict into.
        device: Optional device to load the tensors onto.
        
    Returns:
        A dictionary containing the loaded checkpoint data, including
        any metadata, epoch, and step information.
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If there is an issue loading the checkpoint.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
        
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    return checkpoint
