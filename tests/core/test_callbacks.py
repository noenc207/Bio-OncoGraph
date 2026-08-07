"""
Unit tests for Callback, CallbackPriority, and CallbackManager.

Covers: callback no-op defaults, custom hooks, add/remove/clear,
duplicate detection, dispatch with kwargs, priority ordering,
insertion-order stability, exception isolation, all 13 hook methods,
lifecycle simulation, non-Callback rejection, empty dispatch.
"""

from typing import Any, Dict, List

import pytest

from bio_oncograph.core.callbacks import (
    Callback,
    CallbackManager,
    CallbackPriority,
)


# ── Test Callbacks ──────────────────────────────────────────────────


class RecordingCallback(Callback):
    """Records every dispatched event name and kwargs."""

    def __init__(self, name: str = "recorder") -> None:
        super().__init__()
        self.log: List[str] = []
        self._name = name

    def on_fit_start(self, **kwargs: Any) -> None:
        self.log.append("on_fit_start")

    def on_fit_end(self, **kwargs: Any) -> None:
        self.log.append("on_fit_end")

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.log.append(f"on_epoch_start:{kwargs.get('epoch', '')}")

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.log.append(f"on_epoch_end:{kwargs.get('epoch', '')}")

    def on_batch_start(self, **kwargs: Any) -> None:
        self.log.append("on_batch_start")

    def on_batch_end(self, **kwargs: Any) -> None:
        self.log.append("on_batch_end")

    def on_validation_start(self, **kwargs: Any) -> None:
        self.log.append("on_validation_start")

    def on_validation_end(self, **kwargs: Any) -> None:
        self.log.append("on_validation_end")

    def on_test_start(self, **kwargs: Any) -> None:
        self.log.append("on_test_start")

    def on_test_end(self, **kwargs: Any) -> None:
        self.log.append("on_test_end")

    def on_predict_start(self, **kwargs: Any) -> None:
        self.log.append("on_predict_start")

    def on_predict_end(self, **kwargs: Any) -> None:
        self.log.append("on_predict_end")

    def on_exception(self, **kwargs: Any) -> None:
        self.log.append("on_exception")


class HighPriorityCallback(Callback):
    """Callback with HIGH priority."""

    priority = CallbackPriority.HIGH

    def __init__(self) -> None:
        super().__init__()
        self.log: List[str] = []

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.log.append("high")


class LowPriorityCallback(Callback):
    """Callback with LOW priority."""

    priority = CallbackPriority.LOW

    def __init__(self) -> None:
        super().__init__()
        self.log: List[str] = []

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.log.append("low")


class NormalPriorityCallback(Callback):
    """Callback with NORMAL priority (default)."""

    def __init__(self) -> None:
        super().__init__()
        self.log: List[str] = []

    def on_epoch_start(self, **kwargs: Any) -> None:
        self.log.append("normal")


class FailingCallback(Callback):
    """Callback that raises on on_epoch_start."""

    def on_epoch_start(self, **kwargs: Any) -> None:
        raise RuntimeError("callback crashed")


class KwargsCapture(Callback):
    """Captures kwargs from dispatch."""

    def __init__(self) -> None:
        super().__init__()
        self.captured: Dict[str, Any] = {}

    def on_epoch_end(self, **kwargs: Any) -> None:
        self.captured = dict(kwargs)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def manager() -> CallbackManager:
    return CallbackManager()


# ══════════════════════════════════════════════════════════════════
# Callback base
# ══════════════════════════════════════════════════════════════════


def test_callback_no_op_defaults() -> None:
    """All hooks are no-ops by default (no crash)."""
    cb = Callback()
    cb.on_fit_start()
    cb.on_fit_end()
    cb.on_epoch_start()
    cb.on_epoch_end()
    cb.on_batch_start()
    cb.on_batch_end()
    cb.on_validation_start()
    cb.on_validation_end()
    cb.on_test_start()
    cb.on_test_end()
    cb.on_predict_start()
    cb.on_predict_end()
    cb.on_exception()


def test_callback_default_priority() -> None:
    """Default priority is NORMAL."""
    assert Callback.priority == CallbackPriority.NORMAL


def test_callback_has_all_13_hooks() -> None:
    """Callback defines exactly the 13 required hooks."""
    expected = {
        "on_fit_start", "on_fit_end",
        "on_epoch_start", "on_epoch_end",
        "on_batch_start", "on_batch_end",
        "on_validation_start", "on_validation_end",
        "on_test_start", "on_test_end",
        "on_predict_start", "on_predict_end",
        "on_exception",
    }
    hooks = {
        name for name in dir(Callback)
        if name.startswith("on_") and callable(getattr(Callback, name))
    }
    assert expected <= hooks


# ══════════════════════════════════════════════════════════════════
# CallbackPriority
# ══════════════════════════════════════════════════════════════════


def test_priority_ordering() -> None:
    """HIGH < NORMAL < LOW numerically."""
    assert CallbackPriority.HIGH < CallbackPriority.NORMAL
    assert CallbackPriority.NORMAL < CallbackPriority.LOW


# ══════════════════════════════════════════════════════════════════
# CallbackManager — add / remove / clear
# ══════════════════════════════════════════════════════════════════


def test_add_callback(manager: CallbackManager) -> None:
    """add() registers a callback."""
    cb = RecordingCallback()
    manager.add(cb)
    assert len(manager) == 1


def test_add_multiple(manager: CallbackManager) -> None:
    """Multiple callbacks can be registered."""
    manager.add(RecordingCallback("a"))
    manager.add(RecordingCallback("b"))
    assert len(manager) == 2


def test_add_duplicate_raises(manager: CallbackManager) -> None:
    """Adding the same instance twice raises ValueError."""
    cb = RecordingCallback()
    manager.add(cb)
    with pytest.raises(ValueError, match="already registered"):
        manager.add(cb)


def test_add_non_callback_raises(manager: CallbackManager) -> None:
    """Adding a non-Callback raises TypeError."""
    with pytest.raises(TypeError, match="Expected Callback"):
        manager.add("not_a_callback")  # type: ignore[arg-type]


def test_remove_callback(manager: CallbackManager) -> None:
    """remove() deletes by identity."""
    cb = RecordingCallback()
    manager.add(cb)
    manager.remove(cb)
    assert len(manager) == 0


def test_remove_not_found_raises(
    manager: CallbackManager,
) -> None:
    """remove() on unregistered callback raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        manager.remove(RecordingCallback())


def test_clear(manager: CallbackManager) -> None:
    """clear() removes all callbacks."""
    manager.add(RecordingCallback("a"))
    manager.add(RecordingCallback("b"))
    manager.clear()
    assert len(manager) == 0


# ══════════════════════════════════════════════════════════════════
# CallbackManager — dispatch
# ══════════════════════════════════════════════════════════════════


def test_dispatch_invokes_hook(manager: CallbackManager) -> None:
    """dispatch calls the named hook."""
    cb = RecordingCallback()
    manager.add(cb)
    manager.dispatch("on_fit_start")
    assert "on_fit_start" in cb.log


def test_dispatch_forwards_kwargs(
    manager: CallbackManager,
) -> None:
    """dispatch forwards keyword arguments."""
    cb = KwargsCapture()
    manager.add(cb)
    manager.dispatch("on_epoch_end", epoch=3, loss=0.5)
    assert cb.captured["epoch"] == 3
    assert cb.captured["loss"] == 0.5


def test_dispatch_with_kwargs_in_hook(
    manager: CallbackManager,
) -> None:
    """Epoch number appears in the recording."""
    cb = RecordingCallback()
    manager.add(cb)
    manager.dispatch("on_epoch_start", epoch=7)
    assert "on_epoch_start:7" in cb.log


def test_dispatch_unknown_event(
    manager: CallbackManager,
) -> None:
    """Dispatching an unknown event does not raise."""
    cb = RecordingCallback()
    manager.add(cb)
    manager.dispatch("on_nonexistent_event")  # should not crash


def test_dispatch_all_13_hooks(
    manager: CallbackManager,
) -> None:
    """All 13 hooks can be dispatched."""
    cb = RecordingCallback()
    manager.add(cb)
    hooks = [
        "on_fit_start", "on_fit_end",
        "on_epoch_start", "on_epoch_end",
        "on_batch_start", "on_batch_end",
        "on_validation_start", "on_validation_end",
        "on_test_start", "on_test_end",
        "on_predict_start", "on_predict_end",
        "on_exception",
    ]
    for h in hooks:
        manager.dispatch(h)
    assert len(cb.log) == 13


# ══════════════════════════════════════════════════════════════════
# Priority & ordering
# ══════════════════════════════════════════════════════════════════


def test_priority_dispatch_order(
    manager: CallbackManager,
) -> None:
    """HIGH callbacks fire before NORMAL before LOW."""
    order: List[str] = []

    class H(Callback):
        priority = CallbackPriority.HIGH
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("H")

    class N(Callback):
        priority = CallbackPriority.NORMAL
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("N")

    class L(Callback):
        priority = CallbackPriority.LOW
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("L")

    # Register in reverse order
    manager.add(L())
    manager.add(N())
    manager.add(H())
    manager.dispatch("on_epoch_start")
    assert order == ["H", "N", "L"]


def test_insertion_order_within_same_priority(
    manager: CallbackManager,
) -> None:
    """Same-priority callbacks fire in insertion order."""
    order: List[str] = []

    class A(Callback):
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("A")

    class B(Callback):
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("B")

    class C(Callback):
        def on_epoch_start(self, **kw: Any) -> None:
            order.append("C")

    manager.add(A())
    manager.add(B())
    manager.add(C())
    manager.dispatch("on_epoch_start")
    assert order == ["A", "B", "C"]


def test_deterministic_ordering(
    manager: CallbackManager,
) -> None:
    """Repeated dispatches produce the same order."""
    order1: List[str] = []
    order2: List[str] = []

    class X(Callback):
        priority = CallbackPriority.HIGH
        def on_epoch_start(self, **kw: Any) -> None:
            order1.append("X") if not order2 else order2.append("X")

    class Y(Callback):
        priority = CallbackPriority.LOW
        def on_epoch_start(self, **kw: Any) -> None:
            order1.append("Y") if not order2 else order2.append("Y")

    x, y = X(), Y()
    manager.add(y)
    manager.add(x)
    manager.dispatch("on_epoch_start")
    manager.dispatch("on_epoch_start")
    assert order1 == ["X", "Y"]
    assert order2 == ["X", "Y"]


# ══════════════════════════════════════════════════════════════════
# Exception isolation
# ══════════════════════════════════════════════════════════════════


def test_exception_isolation(
    manager: CallbackManager,
) -> None:
    """Failing callback does not block subsequent callbacks."""
    before = RecordingCallback("before")
    failing = FailingCallback()
    after = RecordingCallback("after")

    # before is NORMAL, failing is NORMAL, after is NORMAL
    # insertion order: before → failing → after
    manager.add(before)
    manager.add(failing)
    manager.add(after)
    manager.dispatch("on_epoch_start")

    assert "on_epoch_start:" in before.log[0]
    assert "on_epoch_start:" in after.log[0]


# ══════════════════════════════════════════════════════════════════
# list_callbacks
# ══════════════════════════════════════════════════════════════════


def test_list_callbacks(manager: CallbackManager) -> None:
    """list_callbacks returns class names in order."""
    manager.add(HighPriorityCallback())
    manager.add(LowPriorityCallback())
    names = manager.list_callbacks()
    assert names[0] == "HighPriorityCallback"
    assert names[1] == "LowPriorityCallback"


def test_list_callbacks_empty(manager: CallbackManager) -> None:
    """list_callbacks returns empty list when no callbacks."""
    assert manager.list_callbacks() == []


# ══════════════════════════════════════════════════════════════════
# Lifecycle simulation
# ══════════════════════════════════════════════════════════════════


def test_full_training_lifecycle(
    manager: CallbackManager,
) -> None:
    """Simulate a full training loop lifecycle."""
    cb = RecordingCallback()
    manager.add(cb)

    manager.dispatch("on_fit_start")
    for epoch in range(2):
        manager.dispatch("on_epoch_start", epoch=epoch)
        manager.dispatch("on_batch_start", batch=0)
        manager.dispatch("on_batch_end", batch=0)
        manager.dispatch("on_validation_start")
        manager.dispatch("on_validation_end")
        manager.dispatch("on_epoch_end", epoch=epoch)
    manager.dispatch("on_fit_end")

    assert cb.log[0] == "on_fit_start"
    assert cb.log[-1] == "on_fit_end"
    # 2 epochs × 5 hooks + fit_start + fit_end = 12
    assert len(cb.log) == 12


def test_repr(manager: CallbackManager) -> None:
    """Manager repr includes callback names."""
    manager.add(RecordingCallback())
    r = repr(manager)
    assert "RecordingCallback" in r
