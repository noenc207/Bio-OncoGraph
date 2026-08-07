"""Validation logic."""
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bio_oncograph.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class Validator:
    """Runs evaluation on the validation dataset."""

    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        metrics_tracker: MetricTracker,
    ) -> Dict[str, float]:
        """Evaluates the model and computes metrics.

        Args:
            model: The neural network model.
            dataloader: Validation DataLoader.
            criterion: The loss function.
            device: Target device.
            metrics_tracker: Object to compute evaluation metrics.

        Returns:
            Dictionary of computed metrics.
        """
        model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        metrics_tracker.reset()

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
                    
                    metrics_tracker.update(outputs, targets)
                else:
                    raise ValueError("Batch must be a tuple/list of at least (features, targets)")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        results = metrics_tracker.compute()
        results["val_loss"] = avg_loss
        
        logger.info(f"Validation Loss: {avg_loss:.4f} - Metrics: {results}")
        return results
