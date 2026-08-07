"""Training engine orchestrating the training process."""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bio_oncograph.training.callbacks import Callback
from bio_oncograph.training.checkpoint import CheckpointManager
from bio_oncograph.training.config import TrainingConfig
from bio_oncograph.training.loop import TrainingLoop
from bio_oncograph.training.validator import Validator
from bio_oncograph.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Facade for managing the entire training lifecycle."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        callbacks: Optional[List[Callback]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initializes the TrainingEngine.

        Args:
            config: Configuration object.
            model: The neural network model.
            optimizer: The optimizer.
            criterion: The loss function.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            scheduler: Learning rate scheduler.
            callbacks: List of callbacks to execute during training.
            device: Device to run the training on.
        """
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        self.loop = TrainingLoop(
            gradient_accumulation_steps=self.config.gradient_accumulation,
            gradient_clip_val=self.config.gradient_clip,
        )
        self.validator = Validator()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
        self.metrics = MetricTracker()
        
        self.scaler = torch.amp.GradScaler(self.device.type) if self.config.mixed_precision else None
        self.current_epoch = 0

    def fit(self) -> None:
        """Executes the training and validation loops for all epochs."""
        logger.info(f"Starting training for {self.config.epochs} epochs on {self.device}")
        
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            
            for cb in self.callbacks:
                cb.on_epoch_begin(self)

            train_metrics = self.loop.train_epoch(
                model=self.model,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                scaler=self.scaler,
            )
            
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.validate_every == 0:
                val_metrics = self.validator.validate(
                    model=self.model,
                    dataloader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                    metrics_tracker=self.metrics,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            for cb in self.callbacks:
                cb.on_epoch_end(self, train_metrics, val_metrics)

        for cb in self.callbacks:
            cb.on_train_end(self)
