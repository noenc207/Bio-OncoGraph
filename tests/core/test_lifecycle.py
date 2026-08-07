"""
Unit tests for LifecycleManager.

Covers: initialize, start, stop, shutdown, event emission for each
transition, error handling with ErrorOccurred emission, auto-stop on
shutdown, lifecycle tracking, components without lifecycle methods.
"""

from typing import Any, List

import pytest

from bio_oncograph.core.events import (
    ComponentInitialized,
    ComponentInitializing,
    ComponentStarted,
    ComponentStarting,
    ComponentStopped,
    ComponentStopping,
    ErrorOccurred,
    Event,
    EventBus,
)
from bio_oncograph.core.lifecycle import LifecycleManager


# ── Helpers ─────────────────────────────────────────────────────────

_event_log: List[str] = []


def _make_recorder(tag: str):  # type: ignore[no-untyped-def]
    """Create a listener that records events to _event_log."""

    def listener(e: Event) -> None:
        _event_log.append(f"{tag}:{e.name}")

    return listener


class FullLifecycleComponent:
    """Component with all lifecycle methods."""

    def __init__(self) -> None:
        self.log: List[str] = []

    def initialize(self) -> None:
        self.log.append("initialized")

    def start(self) -> None:
        self.log.append("started")

    def stop(self) -> None:
        self.log.append("stopped")

    def shutdown(self) -> None:
        self.log.append("shut_down")


class NoLifecycleComponent:
    """Component without any lifecycle methods."""

    pass


class BrokenComponent:
    """Component whose initialize raises."""

    def initialize(self) -> None:
        raise RuntimeError("init failed")


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_log() -> Any:
    """Clear event log between tests."""
    _event_log.clear()
    yield
    _event_log.clear()


@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus."""
    return EventBus()


@pytest.fixture()
def manager(bus: EventBus) -> LifecycleManager:
    """Fresh LifecycleManager."""
    return LifecycleManager(bus)


# ── initialize ──────────────────────────────────────────────────────


def test_initialize_calls_method(
    manager: LifecycleManager,
) -> None:
    """initialize() calls component.initialize()."""
    comp = FullLifecycleComponent()
    manager.initialize(comp, "comp")
    assert "initialized" in comp.log


def test_initialize_emits_events(
    bus: EventBus, manager: LifecycleManager
) -> None:
    """initialize emits Initializing then Initialized events."""
    recorder = _make_recorder("init")
    bus.subscribe(ComponentInitializing, recorder)
    bus.subscribe(ComponentInitialized, recorder)

    manager.initialize(FullLifecycleComponent(), "comp")
    assert _event_log == [
        "init:component.initializing",
        "init:component.initialized",
    ]


def test_initialize_tracks_component(
    manager: LifecycleManager,
) -> None:
    """Component name is added to initialized list."""
    manager.initialize(FullLifecycleComponent(), "my_comp")
    assert "my_comp" in manager.initialized_components


def test_initialize_no_method(
    manager: LifecycleManager,
) -> None:
    """Component without initialize() still completes."""
    comp = NoLifecycleComponent()
    manager.initialize(comp, "no_lc")
    assert "no_lc" in manager.initialized_components


def test_initialize_default_name(
    manager: LifecycleManager,
) -> None:
    """Default name uses the class name."""
    manager.initialize(FullLifecycleComponent())
    assert "FullLifecycleComponent" in manager.initialized_components


# ── start ───────────────────────────────────────────────────────────


def test_start_calls_method(
    manager: LifecycleManager,
) -> None:
    """start() calls component.start()."""
    comp = FullLifecycleComponent()
    manager.start(comp, "comp")
    assert "started" in comp.log


def test_start_emits_events(
    bus: EventBus, manager: LifecycleManager
) -> None:
    """start emits Starting then Started events."""
    recorder = _make_recorder("start")
    bus.subscribe(ComponentStarting, recorder)
    bus.subscribe(ComponentStarted, recorder)

    manager.start(FullLifecycleComponent(), "comp")
    assert _event_log == [
        "start:component.starting",
        "start:component.started",
    ]


def test_start_tracks_component(
    manager: LifecycleManager,
) -> None:
    """Component is added to started list."""
    manager.start(FullLifecycleComponent(), "s")
    assert "s" in manager.started_components


# ── stop ────────────────────────────────────────────────────────────


def test_stop_calls_method(
    manager: LifecycleManager,
) -> None:
    """stop() calls component.stop()."""
    comp = FullLifecycleComponent()
    manager.start(comp, "comp")
    manager.stop(comp, "comp")
    assert "stopped" in comp.log


def test_stop_emits_events(
    bus: EventBus, manager: LifecycleManager
) -> None:
    """stop emits Stopping then Stopped events."""
    recorder = _make_recorder("stop")
    bus.subscribe(ComponentStopping, recorder)
    bus.subscribe(ComponentStopped, recorder)

    comp = FullLifecycleComponent()
    manager.start(comp, "comp")
    _event_log.clear()
    manager.stop(comp, "comp")
    assert _event_log == [
        "stop:component.stopping",
        "stop:component.stopped",
    ]


def test_stop_removes_from_started(
    manager: LifecycleManager,
) -> None:
    """stop removes component from started list."""
    comp = FullLifecycleComponent()
    manager.start(comp, "comp")
    manager.stop(comp, "comp")
    assert "comp" not in manager.started_components


# ── shutdown ────────────────────────────────────────────────────────


def test_shutdown_calls_method(
    manager: LifecycleManager,
) -> None:
    """shutdown() calls component.shutdown()."""
    comp = FullLifecycleComponent()
    manager.initialize(comp, "comp")
    manager.shutdown(comp, "comp")
    assert "shut_down" in comp.log


def test_shutdown_removes_from_initialized(
    manager: LifecycleManager,
) -> None:
    """shutdown removes component from initialized list."""
    comp = FullLifecycleComponent()
    manager.initialize(comp, "comp")
    manager.shutdown(comp, "comp")
    assert "comp" not in manager.initialized_components


def test_shutdown_auto_stops_if_started(
    manager: LifecycleManager,
) -> None:
    """shutdown auto-calls stop if component is started."""
    comp = FullLifecycleComponent()
    manager.initialize(comp, "comp")
    manager.start(comp, "comp")
    manager.shutdown(comp, "comp")
    assert "stopped" in comp.log
    assert "shut_down" in comp.log
    assert "comp" not in manager.started_components


# ── Error handling ──────────────────────────────────────────────────


def test_error_emits_error_event(
    bus: EventBus, manager: LifecycleManager
) -> None:
    """ErrorOccurred is published when lifecycle method raises."""
    errors: List[Event] = []
    bus.subscribe(ErrorOccurred, lambda e: errors.append(e))

    with pytest.raises(RuntimeError, match="init failed"):
        manager.initialize(BrokenComponent(), "broken")

    assert len(errors) == 1
    assert errors[0].source == "broken"
    assert errors[0].payload["phase"] == "initialize"


def test_error_reraises_exception(
    manager: LifecycleManager,
) -> None:
    """The original exception is re-raised to the caller."""
    with pytest.raises(RuntimeError, match="init failed"):
        manager.initialize(BrokenComponent(), "broken")


# ── Properties ──────────────────────────────────────────────────────


def test_event_bus_property(
    bus: EventBus, manager: LifecycleManager
) -> None:
    """Manager exposes its event bus."""
    assert manager.event_bus is bus
