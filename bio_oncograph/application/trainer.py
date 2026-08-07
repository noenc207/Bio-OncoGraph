"""
Trainer for Bio-OncoGraph.

Provides:
  - Training-specific events (``TrainingStarted``, ``EpochFinished``, …).
  - ``IMetrics`` and ``ICheckpoint`` abstract interfaces.
  - ``TrainerCallback`` base class for extensible hooks.
  - ``Trainer`` class orchestrating train / validate / predict loops
    using only abstract interfaces from ``core/``.

The Trainer is backend-agnostic at the *framework* level: it never
imports a concrete model, loss, or dataset.  It uses ``BaseModel``,
``BaseLoss``, and the ``EventBus`` from previous milestones.

Design references:
  - Software Architecture § 19 (Training Engine)
  - Software Architecture § 17 (Callback System)
"""

from __future__ import annotations

import logging
import contextlib
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from bio_oncograph.core.base import BaseLoss, BaseModel
from bio_oncograph.core.data_models import BioGraphOutput
from bio_oncograph.core.events import Event, EventBus

logger = logging.getLogger("BioOncoGraph")


# ── Training events ────────────────────────────────────────────────


@dataclass
class TrainingStarted(Event):
    """Emitted when ``Trainer.fit()`` begins."""

    name: str = "training.started"


@dataclass
class TrainingFinished(Event):
    """Emitted when ``Trainer.fit()`` completes."""

    name: str = "training.finished"


@dataclass
class EpochStarted(Event):
    """Emitted at the start of each epoch."""

    name: str = "epoch.started"


@dataclass
class EpochFinished(Event):
    """Emitted at the end of each epoch."""

    name: str = "epoch.finished"


@dataclass
class ValidationStarted(Event):
    """Emitted when a validation pass begins."""

    name: str = "validation.started"


@dataclass
class ValidationFinished(Event):
    """Emitted when a validation pass completes."""

    name: str = "validation.finished"


@dataclass
class PredictionStarted(Event):
    """Emitted when a prediction pass begins."""

    name: str = "prediction.started"


@dataclass
class PredictionFinished(Event):
    """Emitted when a prediction pass completes."""

    name: str = "prediction.finished"


@dataclass
class RuntimeErrorOccurred(Event):
    """Emitted when the trainer catches an unhandled exception."""

    name: str = "runtime.error"


# ── Abstract Interfaces ────────────────────────────────────────────


class IMetrics(ABC):
    """
    Interface for metric aggregation.

    Concrete implementations (c-index, Brier score, …) will be
    provided in a future milestone.
    """

    @abstractmethod
    def update(self, predictions: Any, targets: Any) -> None:
        """
        Accumulate predictions and targets for metric computation.

        Args:
            predictions: Model output (e.g. ``BioGraphOutput``).
            targets: Ground-truth target (e.g. ``SurvivalTarget``).
        """
        ...

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Returns:
            A dictionary mapping metric names to float values.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state for the next epoch."""
        ...


class ICheckpoint(ABC):
    """
    Interface for checkpoint persistence.

    Concrete implementations (local filesystem, cloud, …) will be
    provided in a future milestone.
    """

    @abstractmethod
    def save(self, state: Dict[str, Any], path: str) -> None:
        """
        Persist a state dictionary.

        Args:
            state: Serialisable state dict.
            path: Target path or identifier.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load a state dictionary.

        Args:
            path: Path or identifier of the checkpoint.

        Returns:
            The loaded state dictionary.
        """
        ...

    @abstractmethod
    def latest(self) -> Optional[str]:
        """
        Return the path of the most recent checkpoint.

        Returns:
            Path string, or ``None`` if no checkpoints exist.
        """
        ...


# ── Callback Base Class ────────────────────────────────────────────


class TrainerCallback:
    """
    Base class for trainer callbacks.

    Override any method to inject custom logic at the corresponding
    point in the training loop.  Default implementations are no-ops
    so subclasses need only override the hooks they care about.
    """

    def on_train_start(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after the last epoch."""
        pass

    def on_epoch_start(
        self, trainer: Trainer, epoch: int
    ) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each epoch with aggregated metrics."""
        pass

    def on_validation_start(self, trainer: Trainer) -> None:
        """Called before each validation pass."""
        pass

    def on_validation_end(
        self, trainer: Trainer, metrics: Dict[str, float]
    ) -> None:
        """Called after each validation pass with metrics."""
        pass

    def on_exception(
        self, trainer: Trainer, exception: Exception
    ) -> None:
        """Called when an unhandled exception occurs."""
        pass


# ── Trainer ─────────────────────────────────────────────────────────


class Trainer:
    """
    Training engine orchestrating train / validate / predict loops.

    The Trainer communicates exclusively through abstract interfaces:
    ``BaseModel``, ``BaseLoss``, ``IMetrics``, ``ICheckpoint``, and
    the ``EventBus``.  No concrete model, dataset, or loss class is
    imported.

    Args:
        model: An abstract ``BaseModel`` instance.
        loss_fn: An abstract ``BaseLoss`` instance.
        optimizer: A PyTorch-compatible optimiser (duck-typed via
            ``zero_grad`` / ``step``).
        event_bus: The framework ``EventBus``.
        train_loader: Iterable yielding training batches.
        val_loader: Optional iterable yielding validation batches.
        test_loader: Optional iterable yielding test batches.
        metrics: Optional ``IMetrics`` implementation.
        checkpoint: Optional ``ICheckpoint`` implementation.
        callbacks: List of ``TrainerCallback`` instances.
        max_epochs: Maximum number of training epochs.
        gradient_clip_max_norm: Maximum gradient norm for clipping.
            Set to ``0.0`` to disable.
    """

    def __init__(
        self,
        model: BaseModel,
        loss_fn: BaseLoss,
        optimizer: Any,
        event_bus: EventBus,
        train_loader: Iterable[Any],
        val_loader: Optional[Iterable[Any]] = None,
        test_loader: Optional[Iterable[Any]] = None,
        metrics: Optional[IMetrics] = None,
        checkpoint: Optional[ICheckpoint] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        max_epochs: int = 100,
        gradient_clip_max_norm: float = 0.0,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        sync_ddp: bool = False,
    ) -> None:
        """Initialise the trainer."""
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._event_bus = event_bus
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
        self._metrics = metrics
        self._checkpoint = checkpoint
        self._callbacks = callbacks or []
        self._max_epochs = max_epochs
        self._gradient_clip_max_norm = gradient_clip_max_norm
        self._gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self._use_amp = use_amp
        self._sync_ddp = sync_ddp
        
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

        # Mutable state
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._best_metric: float = float("inf")

    # ── Properties ──────────────────────────────────────────────

    @property
    def model(self) -> BaseModel:
        """Return the model."""
        return self._model

    @property
    def current_epoch(self) -> int:
        """Return the current epoch index."""
        return self._current_epoch

    @property
    def global_step(self) -> int:
        """Return the global optimiser step count."""
        return self._global_step

    # ── Public API ──────────────────────────────────────────────

    def fit(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Iterates over epochs, calling ``train_epoch`` and optionally
        ``validation_epoch`` per epoch.  Publishes training events
        and invokes callbacks at each boundary.

        Returns:
            Summary dict with ``final_epoch`` and ``global_step``.
        """
        self._publish(TrainingStarted(source="Trainer"))
        self._run_callbacks("on_train_start")

        try:
            for epoch in range(self._current_epoch, self._max_epochs):
                self._current_epoch = epoch
                self._publish(
                    EpochStarted(
                        source="Trainer",
                        payload={"epoch": epoch},
                    )
                )
                self._run_callbacks(
                    "on_epoch_start", epoch=epoch
                )

                train_metrics = self.train_epoch()

                val_metrics: Dict[str, float] = {}
                if self._val_loader is not None:
                    val_metrics = self.validation_epoch()

                all_metrics = {**train_metrics, **val_metrics}
                self._publish(
                    EpochFinished(
                        source="Trainer",
                        payload={"epoch": epoch, **all_metrics},
                    )
                )
                self._run_callbacks(
                    "on_epoch_end",
                    epoch=epoch,
                    metrics=all_metrics,
                )
                logger.info(
                    "Epoch %d finished: %s", epoch, all_metrics
                )
        except Exception as exc:
            self._run_callbacks("on_exception", exception=exc)
            self._publish(
                RuntimeErrorOccurred(
                    source="Trainer",
                    payload={"error": str(exc)},
                )
            )
            raise
        finally:
            self._publish(TrainingFinished(source="Trainer"))
            self._run_callbacks("on_train_end")

        return {
            "final_epoch": self._current_epoch,
            "global_step": self._global_step,
        }

    def validate(self) -> Dict[str, float]:
        """
        Run a single validation pass.

        Returns:
            Dictionary of validation metrics.
        """
        return self.validation_epoch()

    def predict(self) -> List[BioGraphOutput]:
        """
        Run inference on the test loader.

        Returns:
            List of ``BioGraphOutput`` predictions.

        Raises:
            ValueError: If no test loader is configured.
        """
        if self._test_loader is None:
            raise ValueError("No test_loader configured.")

        self._publish(PredictionStarted(source="Trainer"))
        self._model.eval()
        predictions: List[BioGraphOutput] = []

        with torch.no_grad():
            for batch in self._test_loader:
                output = self._model(batch)
                predictions.append(output)

        self._publish(
            PredictionFinished(
                source="Trainer",
                payload={"num_predictions": len(predictions)},
            )
        )
        logger.info("Prediction finished: %d batches", len(predictions))
        return predictions

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the test loader.

        Returns:
            Dictionary of test metrics.

        Raises:
            ValueError: If no test loader is configured.
        """
        if self._test_loader is None:
            raise ValueError("No test_loader configured.")
        return self.test_epoch()

    def train_epoch(self) -> Dict[str, float]:
        """
        Execute a single training epoch with AMP, gradient accumulation, and NaN handling.

        Returns:
            Dictionary with at least `train_loss`.
        """
        self._model.train()
        total_loss = 0.0
        num_batches = 0
        self._optimizer.zero_grad()

        for i, batch in enumerate(self._train_loader):
            is_accumulating = (i + 1) % self._gradient_accumulation_steps != 0 and (i + 1) != len(self._train_loader)
            if self._sync_ddp and is_accumulating and hasattr(self._model, "no_sync"):
                sync_context = self._model.no_sync()
            else:
                sync_context = contextlib.nullcontext()

            with sync_context:
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    output = self._model(batch)
                    loss_output = self._loss_fn(output, batch.survival_target)
                    loss = loss_output.total_loss / self._gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss encountered, skipping batch.")
                    continue

                self._scaler.scale(loss).backward()

            if not is_accumulating:
                if self._gradient_clip_max_norm > 0:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self._gradient_clip_max_norm,
                    )
                
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._global_step += 1

            total_loss += loss_output.total_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def validation_epoch(self) -> Dict[str, float]:
        """
        Execute a single validation pass.

        Returns:
            Dictionary with ``val_loss`` and optional metric values.
        """
        if self._val_loader is None:
            return {}

        self._publish(ValidationStarted(source="Trainer"))
        self._run_callbacks("on_validation_start")
        self._model.eval()
        total_loss = 0.0
        num_batches = 0

        if self._metrics is not None:
            self._metrics.reset()

        with torch.no_grad():
            for batch in self._val_loader:
                output = self._model(batch)
                loss_output = self._loss_fn(
                    output, batch.survival_target
                )
                total_loss += loss_output.total_loss.item()
                num_batches += 1

                if self._metrics is not None:
                    self._metrics.update(
                        output, batch.survival_target
                    )

        avg_loss = total_loss / max(num_batches, 1)
        metrics: Dict[str, float] = {"val_loss": avg_loss}

        if self._metrics is not None:
            metrics.update(self._metrics.compute())

        self._publish(
            ValidationFinished(
                source="Trainer", payload=metrics
            )
        )
        self._run_callbacks(
            "on_validation_end", metrics=metrics
        )
        return metrics

    def test_epoch(self) -> Dict[str, float]:
        """
        Execute a single test evaluation pass.

        Returns:
            Dictionary with ``test_loss`` and optional metrics.
        """
        if self._test_loader is None:
            return {}

        self._model.eval()
        total_loss = 0.0
        num_batches = 0

        if self._metrics is not None:
            self._metrics.reset()

        with torch.no_grad():
            for batch in self._test_loader:
                output = self._model(batch)
                loss_output = self._loss_fn(
                    output, batch.survival_target
                )
                total_loss += loss_output.total_loss.item()
                num_batches += 1

                if self._metrics is not None:
                    self._metrics.update(
                        output, batch.survival_target
                    )

        avg_loss = total_loss / max(num_batches, 1)
        metrics: Dict[str, float] = {"test_loss": avg_loss}

        if self._metrics is not None:
            metrics.update(self._metrics.compute())

        return metrics

    def save_state(self, path: str) -> None:
        """
        Save trainer state via the checkpoint interface.

        Args:
            path: Destination path or identifier.

        Raises:
            ValueError: If no checkpoint handler is configured.
        """
        if self._checkpoint is None:
            raise ValueError("No checkpoint handler configured.")

        state: Dict[str, Any] = {
            "epoch": self._current_epoch,
            "global_step": self._global_step,
            "best_metric": self._best_metric,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        self._checkpoint.save(state, path)
        logger.info("State saved to %s", path)

    def load_state(self, path: str) -> None:
        """
        Load trainer state via the checkpoint interface.

        Args:
            path: Source path or identifier.

        Raises:
            ValueError: If no checkpoint handler is configured.
        """
        if self._checkpoint is None:
            raise ValueError("No checkpoint handler configured.")

        state = self._checkpoint.load(path)
        self._current_epoch = int(state.get("epoch", 0))
        self._global_step = int(state.get("global_step", 0))
        self._best_metric = float(
            state.get("best_metric", float("inf"))
        )
        self._model.load_state_dict(state["model_state_dict"])
        self._optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )
        logger.info("State loaded from %s", path)

    # ── Private helpers ─────────────────────────────────────────

    def _publish(self, event: Event) -> None:
        """Publish an event through the event bus."""
        self._event_bus.publish(event)

    def _run_callbacks(
        self, hook_name: str, **kwargs: Any
    ) -> None:
        """Invoke a named hook on all registered callbacks."""
        for cb in self._callbacks:
            method = getattr(cb, hook_name, None)
            if method is not None and callable(method):
                try:
                    if hook_name in (
                        "on_train_start",
                        "on_train_end",
                        "on_validation_start",
                    ):
                        method(self)
                    elif hook_name == "on_epoch_start":
                        method(self, kwargs["epoch"])
                    elif hook_name == "on_epoch_end":
                        method(
                            self,
                            kwargs["epoch"],
                            kwargs["metrics"],
                        )
                    elif hook_name == "on_validation_end":
                        method(self, kwargs["metrics"])
                    elif hook_name == "on_exception":
                        method(self, kwargs["exception"])
                except Exception:
                    logger.exception(
                        "Error in callback %s.%s",
                        type(cb).__name__,
                        hook_name,
                    )
