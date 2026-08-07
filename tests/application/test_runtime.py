"""
Unit tests for RuntimeEngine and state machine.

Covers: state transitions (valid and invalid), initialize/shutdown/
reset lifecycle, run in all modes, error state recovery, duplicate
initialize, shutdown from various states, missing trainer, event
publishing on error, trainer reuse after reset, and deterministic
state progression.
"""

from typing import Any, Dict, List, Optional

import pytest
import torch

from bio_oncograph.core.base import BaseLoss, BaseModel
from bio_oncograph.core.data_models import BioGraphOutput, SurvivalTarget
from bio_oncograph.core.events import Event, EventBus
from bio_oncograph.core.typing import LossOutput

from bio_oncograph.application.runtime import (
    InvalidStateTransitionError,
    RuntimeEngine,
    RuntimeState,
    _VALID_TRANSITIONS,
)
from bio_oncograph.application.trainer import (
    RuntimeErrorOccurred,
    Trainer,
)


# ── Stubs (same lightweight stubs used in test_trainer) ─────────────


class StubModel(BaseModel):
    """Minimal model returning fixed outputs."""

    def __init__(self) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, data: Any) -> BioGraphOutput:
        return BioGraphOutput(
            hazard_logits=torch.zeros(1, 4),
            survival_curves=torch.ones(1, 4),
        )


class StubLoss(BaseLoss):
    """Minimal loss returning a fixed differentiable scalar."""

    def compute(
        self, pred: BioGraphOutput, target: SurvivalTarget
    ) -> LossOutput:
        loss = (pred.hazard_logits.sum() * 0 + 0.5).requires_grad_(
            True
        )
        return LossOutput(
            total_loss=loss, nll=loss, ranking=loss, topo=loss
        )


class StubBatch:
    """Minimal batch with a survival_target attribute."""

    def __init__(self) -> None:
        self.survival_target = SurvivalTarget(
            time_bins=torch.tensor([1]),
            event_indicators=torch.tensor([True]),
        )


class FailModel(BaseModel):
    """Model that raises on forward."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, data: Any) -> BioGraphOutput:
        raise RuntimeError("model exploded")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus."""
    return EventBus()


@pytest.fixture()
def model() -> StubModel:
    return StubModel()


@pytest.fixture()
def loss_fn() -> StubLoss:
    return StubLoss()


@pytest.fixture()
def engine(bus: EventBus) -> RuntimeEngine:
    """Fresh RuntimeEngine."""
    return RuntimeEngine(bus)


@pytest.fixture()
def trainer(
    model: StubModel,
    loss_fn: StubLoss,
    bus: EventBus,
) -> Trainer:
    """Trainer with 1-epoch training and test loaders."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch(), StubBatch()],
        val_loader=[StubBatch()],
        test_loader=[StubBatch()],
        max_epochs=1,
    )


# ── Initial state ──────────────────────────────────────────────────


def test_initial_state(engine: RuntimeEngine) -> None:
    """Engine starts in CREATED state."""
    assert engine.state == RuntimeState.CREATED


def test_trainer_initially_none(engine: RuntimeEngine) -> None:
    """Trainer is None before initialize."""
    assert engine.trainer is None


# ── initialize ──────────────────────────────────────────────────────


def test_initialize_transitions_to_initialized(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """initialize() transitions CREATED → INITIALIZED."""
    engine.initialize(trainer)
    assert engine.state == RuntimeState.INITIALIZED
    assert engine.trainer is trainer


def test_duplicate_initialize_raises(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """initialize() from INITIALIZED raises."""
    engine.initialize(trainer)
    with pytest.raises(InvalidStateTransitionError):
        engine.initialize(trainer)


# ── run: train mode ─────────────────────────────────────────────────


def test_run_train(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """run('train') executes fit and finishes."""
    engine.initialize(trainer)
    result = engine.run("train")
    assert "final_epoch" in result
    assert engine.state == RuntimeState.FINISHED


def test_run_train_case_insensitive(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """Mode string is case-insensitive."""
    engine.initialize(trainer)
    result = engine.run("TRAIN")
    assert "final_epoch" in result


# ── run: validate mode ──────────────────────────────────────────────


def test_run_validate(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """run('validate') executes validate and finishes."""
    engine.initialize(trainer)
    result = engine.run("validate")
    assert "val_loss" in result
    assert engine.state == RuntimeState.FINISHED


# ── run: test mode ──────────────────────────────────────────────────


def test_run_test(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """run('test') executes evaluate and finishes."""
    engine.initialize(trainer)
    result = engine.run("test")
    assert "test_loss" in result
    assert engine.state == RuntimeState.FINISHED


# ── run: predict mode ──────────────────────────────────────────────


def test_run_predict(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """run('predict') returns predictions list."""
    engine.initialize(trainer)
    result = engine.run("predict")
    assert "predictions" in result
    assert len(result["predictions"]) == 1
    assert engine.state == RuntimeState.FINISHED


# ── run: error cases ────────────────────────────────────────────────


def test_run_unknown_mode_raises(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """Unknown mode raises ValueError."""
    engine.initialize(trainer)
    with pytest.raises(ValueError, match="Unknown mode"):
        engine.run("nonexistent")


def test_run_without_initialize_raises(
    engine: RuntimeEngine,
) -> None:
    """run() before initialize() raises ValueError."""
    # Force to INITIALIZED state without trainer
    engine._state = RuntimeState.INITIALIZED
    with pytest.raises(ValueError, match="not initialized"):
        engine.run("train")


def test_run_from_created_raises(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """run() from CREATED raises InvalidStateTransitionError."""
    with pytest.raises(
        (InvalidStateTransitionError, ValueError)
    ):
        engine.run("train")


# ── Error state ─────────────────────────────────────────────────────


def test_error_state_on_exception(bus: EventBus) -> None:
    """Runtime transitions to ERROR when trainer raises."""
    fail_model = FailModel()
    loss_fn = StubLoss()
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.01
    )
    t = Trainer(
        model=fail_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch()],
        max_epochs=1,
    )
    engine = RuntimeEngine(bus)
    engine.initialize(t)
    with pytest.raises(RuntimeError, match="model exploded"):
        engine.run("train")
    assert engine.state == RuntimeState.ERROR


def test_error_publishes_event(bus: EventBus) -> None:
    """Error during run publishes RuntimeErrorOccurred."""
    errors: List[Event] = []
    bus.subscribe(
        RuntimeErrorOccurred, lambda e: errors.append(e)
    )

    fail_model = FailModel()
    optimizer = torch.optim.SGD(
        [torch.nn.Parameter(torch.zeros(1))], lr=0.01
    )
    t = Trainer(
        model=fail_model,
        loss_fn=StubLoss(),
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch()],
        max_epochs=1,
    )
    engine = RuntimeEngine(bus)
    engine.initialize(t)
    with pytest.raises(RuntimeError):
        engine.run("train")
    # At least one RuntimeErrorOccurred from trainer or engine
    assert len(errors) >= 1


def test_error_recovery_via_initialize(
    bus: EventBus, trainer: Trainer
) -> None:
    """ERROR state can transition back to INITIALIZED."""
    engine = RuntimeEngine(bus)
    engine._state = RuntimeState.ERROR
    engine.initialize(trainer)
    assert engine.state == RuntimeState.INITIALIZED


# ── shutdown ────────────────────────────────────────────────────────


def test_shutdown_from_initialized(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """shutdown from INITIALIZED transitions to STOPPED."""
    engine.initialize(trainer)
    engine.shutdown()
    assert engine.state == RuntimeState.STOPPED
    assert engine.trainer is None


def test_shutdown_from_finished(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """shutdown from FINISHED transitions to STOPPED."""
    engine.initialize(trainer)
    engine.run("train")
    engine.shutdown()
    assert engine.state == RuntimeState.STOPPED


def test_shutdown_from_created_raises(
    engine: RuntimeEngine,
) -> None:
    """shutdown from CREATED raises (not in valid transitions)."""
    with pytest.raises(InvalidStateTransitionError):
        engine.shutdown()


# ── reset ───────────────────────────────────────────────────────────


def test_reset_from_stopped(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """reset from STOPPED transitions to CREATED."""
    engine.initialize(trainer)
    engine.shutdown()
    engine.reset()
    assert engine.state == RuntimeState.CREATED
    assert engine.trainer is None


def test_reset_from_finished(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """reset from FINISHED transitions to CREATED."""
    engine.initialize(trainer)
    engine.run("train")
    engine.reset()
    assert engine.state == RuntimeState.CREATED


def test_reset_from_initialized_raises(
    engine: RuntimeEngine, trainer: Trainer
) -> None:
    """reset from INITIALIZED raises."""
    engine.initialize(trainer)
    with pytest.raises(InvalidStateTransitionError):
        engine.reset()


# ── Invalid state transitions ──────────────────────────────────────


def test_invalid_transition_created_to_training(
    engine: RuntimeEngine,
) -> None:
    """CREATED → TRAINING is not valid."""
    with pytest.raises(InvalidStateTransitionError):
        engine._transition(RuntimeState.TRAINING)


def test_invalid_transition_finished_to_training(
    engine: RuntimeEngine,
) -> None:
    """FINISHED → TRAINING is not valid (must reinitialise)."""
    engine._state = RuntimeState.FINISHED
    with pytest.raises(InvalidStateTransitionError):
        engine._transition(RuntimeState.TRAINING)


# ── State machine completeness ──────────────────────────────────────


def test_all_states_have_transitions() -> None:
    """Every RuntimeState has an entry in the transition table."""
    for state in RuntimeState:
        assert state in _VALID_TRANSITIONS, (
            f"{state.name} missing from transition table"
        )


# ── Trainer reuse after reset ───────────────────────────────────────


def test_full_lifecycle_reuse(
    bus: EventBus,
    model: StubModel,
    loss_fn: StubLoss,
) -> None:
    """Full cycle: init → run → shutdown → reset → init → run."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    t = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        event_bus=bus,
        train_loader=[StubBatch()],
        max_epochs=1,
    )
    engine = RuntimeEngine(bus)

    # First cycle
    engine.initialize(t)
    engine.run("train")
    assert engine.state == RuntimeState.FINISHED
    engine.shutdown()
    engine.reset()
    assert engine.state == RuntimeState.CREATED

    # Second cycle with same trainer
    engine.initialize(t)
    result = engine.run("validate")
    assert engine.state == RuntimeState.FINISHED


# ── Properties ──────────────────────────────────────────────────────


def test_event_bus_property(
    engine: RuntimeEngine, bus: EventBus
) -> None:
    """Engine exposes its EventBus."""
    assert engine.event_bus is bus
