"""
Unit tests for Trainer.

Uses lightweight stub implementations of BaseModel, BaseLoss,
IMetrics, ICheckpoint to test orchestration logic without
requiring real neural network computation.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from bio_oncograph.core.base import BaseLoss, BaseModel
from bio_oncograph.core.data_models import BioGraphOutput, SurvivalTarget
from bio_oncograph.core.events import Event, EventBus
from bio_oncograph.core.typing import LossOutput

from bio_oncograph.application.trainer import (
    EpochFinished,
    EpochStarted,
    ICheckpoint,
    IMetrics,
    PredictionFinished,
    PredictionStarted,
    RuntimeErrorOccurred,
    Trainer,
    TrainerCallback,
    TrainingFinished,
    TrainingStarted,
    ValidationFinished,
    ValidationStarted,
)


# ── Stubs ───────────────────────────────────────────────────────────


class StubModel(BaseModel):
    """Model returning fixed zero outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data: Any) -> BioGraphOutput:
        return BioGraphOutput(
            hazard_logits=torch.zeros(1, 4),
            survival_curves=torch.ones(1, 4),
        )


class StubLoss(BaseLoss):
    """Loss returning a fixed differentiable scalar."""

    def compute(
        self, pred: BioGraphOutput, target: SurvivalTarget
    ) -> LossOutput:
        loss = (pred.hazard_logits.sum() * 0 + 0.5).requires_grad_(
            True
        )
        return LossOutput(
            total_loss=loss, nll=loss, ranking=loss, topo=loss
        )


class StubMetrics(IMetrics):
    """Metrics that count updates."""

    def __init__(self) -> None:
        self._count = 0

    def update(self, predictions: Any, targets: Any) -> None:
        self._count += 1

    def compute(self) -> Dict[str, float]:
        return {"c_index": 0.85, "update_count": float(self._count)}

    def reset(self) -> None:
        self._count = 0


class StubCheckpoint(ICheckpoint):
    """In-memory checkpoint store."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, state: Dict[str, Any], path: str) -> None:
        self._store[path] = state

    def load(self, path: str) -> Dict[str, Any]:
        return self._store[path]

    def latest(self) -> Optional[str]:
        if not self._store:
            return None
        return list(self._store.keys())[-1]


class StubBatch:
    """Minimal batch object with survival_target."""

    def __init__(self) -> None:
        self.survival_target = SurvivalTarget(
            time_bins=torch.tensor([1]),
            event_indicators=torch.tensor([True]),
        )


class RecordingCallback(TrainerCallback):
    """Callback that records hook invocations."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def on_train_start(self, trainer: Trainer) -> None:
        self.calls.append("on_train_start")

    def on_train_end(self, trainer: Trainer) -> None:
        self.calls.append("on_train_end")

    def on_epoch_start(
        self, trainer: Trainer, epoch: int
    ) -> None:
        self.calls.append(f"on_epoch_start:{epoch}")

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        self.calls.append(f"on_epoch_end:{epoch}")

    def on_validation_start(self, trainer: Trainer) -> None:
        self.calls.append("on_validation_start")

    def on_validation_end(
        self, trainer: Trainer, metrics: Dict[str, float]
    ) -> None:
        self.calls.append("on_validation_end")

    def on_exception(
        self, trainer: Trainer, exception: Exception
    ) -> None:
        self.calls.append(f"on_exception:{type(exception).__name__}")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def model() -> StubModel:
    return StubModel()


@pytest.fixture()
def loss_fn() -> StubLoss:
    return StubLoss()


@pytest.fixture()
def optimizer(model: StubModel) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture()
def train_loader() -> List[StubBatch]:
    return [StubBatch(), StubBatch()]


@pytest.fixture()
def val_loader() -> List[StubBatch]:
    return [StubBatch()]


@pytest.fixture()
def trainer(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
    train_loader: List[StubBatch],
    val_loader: List[StubBatch],
) -> Trainer:
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=2,
    )


# ── Tests ───────────────────────────────────────────────────────────


def test_fit_runs_epochs(trainer: Trainer) -> None:
    """fit() runs max_epochs and returns summary."""
    result = trainer.fit()
    assert result["final_epoch"] == 1  # 0-indexed
    assert result["global_step"] == 4  # 2 epochs × 2 batches


def test_train_epoch_returns_loss(trainer: Trainer) -> None:
    """train_epoch returns dict with train_loss."""
    metrics = trainer.train_epoch()
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)


def test_validation_epoch_returns_loss(trainer: Trainer) -> None:
    """validation_epoch returns dict with val_loss."""
    metrics = trainer.validation_epoch()
    assert "val_loss" in metrics


def test_predict_returns_outputs(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
) -> None:
    """predict returns a list of BioGraphOutput."""
    test_data = [StubBatch(), StubBatch()]
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[],
        test_loader=test_data,
        max_epochs=1,
    )
    preds = t.predict()
    assert len(preds) == 2
    assert isinstance(preds[0], BioGraphOutput)


def test_predict_no_loader_raises(trainer: Trainer) -> None:
    """predict without test_loader raises ValueError."""
    with pytest.raises(ValueError, match="test_loader"):
        trainer.predict()


def test_evaluate_returns_metrics(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
) -> None:
    """evaluate returns dict with test_loss."""
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[],
        test_loader=[StubBatch()],
        max_epochs=1,
    )
    metrics = t.evaluate()
    assert "test_loss" in metrics


def test_callbacks_invoked(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
    train_loader: List[StubBatch],
    val_loader: List[StubBatch],
) -> None:
    """All callback hooks are invoked during fit()."""
    cb = RecordingCallback()
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=[cb],
        max_epochs=1,
    )
    t.fit()
    assert "on_train_start" in cb.calls
    assert "on_epoch_start:0" in cb.calls
    assert "on_validation_start" in cb.calls
    assert "on_validation_end" in cb.calls
    assert "on_epoch_end:0" in cb.calls
    assert "on_train_end" in cb.calls


def test_event_publishing(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
    train_loader: List[StubBatch],
) -> None:
    """Trainer publishes training events during fit()."""
    published: List[str] = []
    bus.subscribe(
        TrainingStarted, lambda e: published.append(e.name)
    )
    bus.subscribe(
        EpochStarted, lambda e: published.append(e.name)
    )
    bus.subscribe(
        EpochFinished, lambda e: published.append(e.name)
    )
    bus.subscribe(
        TrainingFinished, lambda e: published.append(e.name)
    )

    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=train_loader,
        max_epochs=1,
    )
    t.fit()
    assert "training.started" in published
    assert "epoch.started" in published
    assert "epoch.finished" in published
    assert "training.finished" in published


def test_gradient_clipping(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
    train_loader: List[StubBatch],
) -> None:
    """Gradient clipping does not crash."""
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=train_loader,
        max_epochs=1,
        gradient_clip_max_norm=1.0,
    )
    t.fit()  # Should not raise


def test_metrics_interface(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
    val_loader: List[StubBatch],
) -> None:
    """Metrics.update/compute/reset are called correctly."""
    metrics = StubMetrics()
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch()],
        val_loader=val_loader,
        metrics=metrics,
        max_epochs=1,
    )
    t.fit()
    # validate was called once with 1 batch
    val_metrics = t.validate()
    assert "c_index" in val_metrics


def test_checkpoint_save_load(
    model: StubModel,
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
) -> None:
    """save_state/load_state round-trips via ICheckpoint."""
    ckpt = StubCheckpoint()
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch()],
        checkpoint=ckpt,
        max_epochs=1,
    )
    t.fit()
    t.save_state("ckpt_001")
    assert ckpt.latest() == "ckpt_001"

    t2 = Trainer(
        model=StubModel(),
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD(
            StubModel().parameters(), lr=0.01
        ),
        event_bus=bus,
        train_loader=[],
        checkpoint=ckpt,
        max_epochs=1,
    )
    t2.load_state("ckpt_001")
    assert t2.current_epoch == t.current_epoch


def test_save_state_no_checkpoint_raises(
    trainer: Trainer,
) -> None:
    """save_state without checkpoint raises ValueError."""
    with pytest.raises(ValueError, match="checkpoint"):
        trainer.save_state("path")


def test_exception_propagation(
    loss_fn: StubLoss,
    optimizer: torch.optim.SGD,
    bus: EventBus,
) -> None:
    """Exceptions during training propagate with events."""
    errors: List[Event] = []
    bus.subscribe(RuntimeErrorOccurred, lambda e: errors.append(e))

    class FailModel(BaseModel):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, data: Any) -> BioGraphOutput:
            raise RuntimeError("forward exploded")

    t = Trainer(
        model=FailModel(),
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01),
        event_bus=bus,
        train_loader=[StubBatch()],
        max_epochs=1,
    )
    with pytest.raises(RuntimeError, match="forward exploded"):
        t.fit()
    assert len(errors) == 1


def test_trainer_properties(trainer: Trainer) -> None:
    """Properties are accessible."""
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0
    assert trainer.model is not None
