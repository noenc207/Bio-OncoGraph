"""Training callbacks."""
import abc
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Callback(abc.ABC):
    """Abstract base class for training callbacks."""

    def on_train_begin(self, engine: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, engine: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, engine: Any) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, engine: Any, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass


class EarlyStoppingCallback(Callback):
    """Callback to stop training when a monitored metric stops improving."""

    def __init__(self, monitor: str = "val_loss", patience: int = 5, mode: str = "min") -> None:
        """Initializes EarlyStoppingCallback.

        Args:
            monitor: The metric to monitor.
            patience: Number of epochs to wait for improvement.
            mode: 'min' or 'max', indicating if the metric should be minimized or maximized.
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, engine: Any, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Checks for early stopping conditions at epoch end."""
        metrics = {**train_metrics, **val_metrics}
        if self.monitor not in metrics:
            logger.warning(f"Metric {self.monitor} not found for EarlyStopping.")
            return

        current_score = metrics[self.monitor]
        
        is_better = current_score < self.best_score if self.mode == "min" else current_score > self.best_score
        
        if is_better:
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered at epoch {engine.current_epoch + 1}.")
                engine.config.epochs = engine.current_epoch + 1
                self.stopped_epoch = engine.current_epoch + 1


class ModelCheckpointCallback(Callback):
    """Callback to save model checkpoints."""

    def __init__(self, monitor: str = "val_loss", mode: str = "min", save_last: bool = True) -> None:
        """Initializes ModelCheckpointCallback.

        Args:
            monitor: The metric to monitor for the 'best' checkpoint.
            mode: 'min' or 'max', depending on the metric.
            save_last: Whether to save the latest checkpoint at each epoch.
        """
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, engine: Any, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Saves checkpoints at the end of an epoch."""
        if self.save_last:
            engine.checkpoint_manager.save(
                filename="latest.pt",
                epoch=engine.current_epoch,
                model=engine.model,
                optimizer=engine.optimizer,
                scheduler=engine.scheduler,
                scaler=engine.scaler,
            )

        if (engine.current_epoch + 1) % engine.config.save_every == 0:
            engine.checkpoint_manager.save(
                filename=f"epoch_{engine.current_epoch + 1:04d}.pt",
                epoch=engine.current_epoch,
                model=engine.model,
                optimizer=engine.optimizer,
                scheduler=engine.scheduler,
                scaler=engine.scaler,
            )

        metrics = {**train_metrics, **val_metrics}
        if self.monitor in metrics:
            current_score = metrics[self.monitor]
            is_better = current_score < self.best_score if self.mode == "min" else current_score > self.best_score
            
            if is_better:
                self.best_score = current_score
                engine.checkpoint_manager.save(
                    filename="best.pt",
                    epoch=engine.current_epoch,
                    model=engine.model,
                    optimizer=engine.optimizer,
                    scheduler=engine.scheduler,
                    scaler=engine.scaler,
                    extras={self.monitor: self.best_score},
                )
                logger.info(f"New best model saved with {self.monitor}: {self.best_score:.4f}")
