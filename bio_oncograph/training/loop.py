"""Standard training loop implementation."""
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bio_oncograph.training.trainer import Trainer

logger = logging.getLogger(__name__)


class TrainingLoop(Trainer):
    """Standard implementation of the training loop.
    
    Handles forward passes, loss computation, backward passes,
    optimizer steps, and optional mixed precision.
    """

    def __init__(self, gradient_accumulation_steps: int = 1, gradient_clip_val: float = 1.0) -> None:
        """Initializes the TrainingLoop.

        Args:
            gradient_accumulation_steps: Number of batches to accumulate gradients.
            gradient_clip_val: Maximum norm for gradient clipping.
        """
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Runs a standard training epoch over the given dataloader.

        Args:
            model: The model to train.
            dataloader: Training dataloader.
            optimizer: The optimizer.
            criterion: The loss function.
            device: Target device.
            scaler: Optional GradScaler for AMP.

        Returns:
            Dictionary with average loss.
        """
        model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                features, targets = batch[0], batch[1]
                if isinstance(features, torch.Tensor):
                    features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                raise ValueError("Batch must be a tuple/list of at least (features, targets)")

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if scaler is not None:
                    if self.gradient_clip_val > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if self.gradient_clip_val > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)

            total_loss += (loss.item() * self.gradient_accumulation_steps)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluates the model.

        Args:
            model: The model to evaluate.
            dataloader: Validation/test dataloader.
            criterion: The loss function.
            device: Target device.

        Returns:
            Dictionary with average loss.
        """
        model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    features, targets = batch[0], batch[1]
                    if isinstance(features, torch.Tensor):
                        features = features.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                else:
                    raise ValueError("Batch must be a tuple/list of at least (features, targets)")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}
