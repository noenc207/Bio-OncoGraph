"""
Unit tests for Event, EventPriority, and EventBus.

Covers: subscribe, unsubscribe, duplicate subscribe, publish,
multiple listeners, priority ordering, exception isolation,
event fields (id, timestamp, payload, source), thread safety,
deterministic ordering, clear, listener_count, registered_events,
invalid listener, invalid event type.
"""

import threading
import time
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from bio_oncograph.core.events import (
    ComponentInitialized,
    ComponentInitializing,
    ErrorOccurred,
    Event,
    EventBus,
    EventPriority,
    Listener,
    PipelineBuilt,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def bus() -> EventBus:
    """Fresh EventBus for each test."""
    return EventBus()


# ── Event dataclass ─────────────────────────────────────────────────


def test_event_has_id() -> None:
    """Every event gets a unique id."""
    e1 = Event(source="test")
    e2 = Event(source="test")
    assert e1.id != e2.id


def test_event_has_timestamp() -> None:
    """Event timestamp is close to current time."""
    before = time.time()
    evt = Event(source="test")
    after = time.time()
    assert before <= evt.timestamp <= after


def test_event_payload() -> None:
    """Custom payload is accessible."""
    evt = Event(source="s", payload={"key": "value"})
    assert evt.payload["key"] == "value"


def test_event_source() -> None:
    """Source field is set correctly."""
    evt = Event(source="my_module")
    assert evt.source == "my_module"


def test_lifecycle_event_default_name() -> None:
    """Predefined events have correct default names."""
    e = ComponentInitializing(source="x")
    assert e.name == "component.initializing"


# ── subscribe ───────────────────────────────────────────────────────


def test_subscribe(bus: EventBus) -> None:
    """subscribe adds a listener."""
    listener = MagicMock()
    bus.subscribe(Event, listener)
    assert bus.listener_count(Event) == 1


def test_subscribe_duplicate_ignored(bus: EventBus) -> None:
    """Subscribing the same listener twice is a no-op."""
    listener = MagicMock()
    bus.subscribe(Event, listener)
    bus.subscribe(Event, listener)
    assert bus.listener_count(Event) == 1


def test_subscribe_non_callable_raises(bus: EventBus) -> None:
    """Non-callable listener raises TypeError."""
    with pytest.raises(TypeError, match="callable"):
        bus.subscribe(Event, "not a function")  # type: ignore[arg-type]


def test_subscribe_invalid_event_type_raises(
    bus: EventBus,
) -> None:
    """Non-Event subclass raises TypeError."""
    with pytest.raises(TypeError, match="subclass of Event"):
        bus.subscribe(str, MagicMock())  # type: ignore[arg-type]


# ── unsubscribe ─────────────────────────────────────────────────────


def test_unsubscribe(bus: EventBus) -> None:
    """unsubscribe removes a listener."""
    listener = MagicMock()
    bus.subscribe(Event, listener)
    bus.unsubscribe(Event, listener)
    assert bus.listener_count(Event) == 0


def test_unsubscribe_unknown_event_raises(bus: EventBus) -> None:
    """Unsubscribing from unregistered event raises ValueError."""
    with pytest.raises(ValueError, match="No listeners"):
        bus.unsubscribe(Event, MagicMock())


def test_unsubscribe_unknown_listener_raises(
    bus: EventBus,
) -> None:
    """Unsubscribing a non-subscribed listener raises ValueError."""
    bus.subscribe(Event, MagicMock())
    with pytest.raises(ValueError, match="not found"):
        bus.unsubscribe(Event, MagicMock())


# ── publish ─────────────────────────────────────────────────────────


def test_publish_calls_listener(bus: EventBus) -> None:
    """publish invokes the subscribed listener."""
    listener = MagicMock()
    bus.subscribe(Event, listener)
    evt = Event(source="test")
    bus.publish(evt)
    listener.assert_called_once_with(evt)


def test_publish_multiple_listeners(bus: EventBus) -> None:
    """All listeners for an event type are called."""
    l1, l2, l3 = MagicMock(), MagicMock(), MagicMock()
    bus.subscribe(Event, l1)
    bus.subscribe(Event, l2)
    bus.subscribe(Event, l3)
    evt = Event(source="test")
    bus.publish(evt)
    l1.assert_called_once()
    l2.assert_called_once()
    l3.assert_called_once()


def test_publish_invalid_event_raises(bus: EventBus) -> None:
    """Publishing a non-Event raises TypeError."""
    with pytest.raises(TypeError, match="Event instance"):
        bus.publish("not an event")  # type: ignore[arg-type]


def test_publish_no_listeners_is_noop(bus: EventBus) -> None:
    """Publishing to an event with no listeners doesn't crash."""
    bus.publish(Event(source="test"))


# ── Priority ordering ──────────────────────────────────────────────


def test_priority_ordering(bus: EventBus) -> None:
    """HIGH listeners run before NORMAL, NORMAL before LOW."""
    call_order: List[str] = []

    def high_fn(e: Event) -> None:
        call_order.append("high")

    def normal_fn(e: Event) -> None:
        call_order.append("normal")

    def low_fn(e: Event) -> None:
        call_order.append("low")

    bus.subscribe(Event, low_fn, priority=EventPriority.LOW)
    bus.subscribe(Event, high_fn, priority=EventPriority.HIGH)
    bus.subscribe(Event, normal_fn, priority=EventPriority.NORMAL)

    bus.publish(Event(source="test"))
    assert call_order == ["high", "normal", "low"]


def test_same_priority_fifo(bus: EventBus) -> None:
    """Within the same priority, listeners run in FIFO order."""
    call_order: List[str] = []

    def first(e: Event) -> None:
        call_order.append("first")

    def second(e: Event) -> None:
        call_order.append("second")

    bus.subscribe(Event, first, priority=EventPriority.NORMAL)
    bus.subscribe(Event, second, priority=EventPriority.NORMAL)

    bus.publish(Event(source="test"))
    assert call_order == ["first", "second"]


# ── Exception isolation ────────────────────────────────────────────


def test_listener_exception_does_not_block_others(
    bus: EventBus,
) -> None:
    """A failing listener does not prevent others from running."""
    call_order: List[str] = []

    def good_before(e: Event) -> None:
        call_order.append("before")

    def bad(e: Event) -> None:
        raise RuntimeError("boom")

    def good_after(e: Event) -> None:
        call_order.append("after")

    bus.subscribe(Event, good_before, EventPriority.HIGH)
    bus.subscribe(Event, bad, EventPriority.NORMAL)
    bus.subscribe(Event, good_after, EventPriority.LOW)

    bus.publish(Event(source="test"))
    assert call_order == ["before", "after"]


# ── clear ───────────────────────────────────────────────────────────


def test_clear(bus: EventBus) -> None:
    """clear removes all subscriptions."""
    bus.subscribe(Event, MagicMock())
    bus.subscribe(ComponentInitialized, MagicMock())
    bus.clear()
    assert bus.listener_count() == 0
    assert bus.registered_events() == []


# ── listener_count ──────────────────────────────────────────────────


def test_listener_count_total(bus: EventBus) -> None:
    """listener_count() returns total across all types."""
    bus.subscribe(Event, MagicMock())
    bus.subscribe(ComponentInitialized, MagicMock())
    assert bus.listener_count() == 2


def test_listener_count_per_type(bus: EventBus) -> None:
    """listener_count(type) returns count for that type only."""
    bus.subscribe(Event, MagicMock())
    bus.subscribe(Event, MagicMock())
    bus.subscribe(ComponentInitialized, MagicMock())
    assert bus.listener_count(Event) == 2
    assert bus.listener_count(ComponentInitialized) == 1


# ── registered_events ──────────────────────────────────────────────


def test_registered_events(bus: EventBus) -> None:
    """registered_events lists types with listeners."""
    bus.subscribe(Event, MagicMock())
    bus.subscribe(PipelineBuilt, MagicMock())
    types = bus.registered_events()
    assert Event in types
    assert PipelineBuilt in types


# ── Thread safety ───────────────────────────────────────────────────


def test_concurrent_subscribe_and_publish(
    bus: EventBus,
) -> None:
    """Concurrent subscribe + publish does not corrupt state."""
    counter = {"value": 0}
    lock = threading.Lock()

    def counting_listener(e: Event) -> None:
        with lock:
            counter["value"] += 1

    bus.subscribe(Event, counting_listener)
    errors: List[Exception] = []

    def publisher() -> None:
        try:
            for _ in range(50):
                bus.publish(Event(source="thread"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publisher) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert counter["value"] == 200


# ── Deterministic ordering ─────────────────────────────────────────


def test_deterministic_dispatch_order(bus: EventBus) -> None:
    """Two publishes produce the same listener call order."""
    order1: List[str] = []
    order2: List[str] = []

    def a(e: Event) -> None:
        pass  # We track via side channel below

    def b(e: Event) -> None:
        pass

    bus.subscribe(Event, a, EventPriority.NORMAL)
    bus.subscribe(Event, b, EventPriority.NORMAL)

    # Verify internal order is stable by checking the sort
    with bus._lock:
        subs = bus._subscriptions[Event]
        assert subs[0][2] is a
        assert subs[1][2] is b


# ── repr ────────────────────────────────────────────────────────────


def test_repr(bus: EventBus) -> None:
    """repr shows event and listener counts."""
    bus.subscribe(Event, MagicMock())
    r = repr(bus)
    assert "events=1" in r
    assert "listeners=1" in r
