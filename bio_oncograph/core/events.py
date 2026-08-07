"""
Event System for Bio-OncoGraph.

Provides:
  - ``Event`` base dataclass with id, timestamp, name, source, payload.
  - ``EventPriority`` enum (HIGH, NORMAL, LOW).
  - ``EventBus`` for thread-safe publish/subscribe with priority dispatch
    and exception isolation.
  - Ten predefined lifecycle event types.

Design references:
  - Software Architecture § 16 (Event System)
  - Software Architecture § 17 (Callback System)
  - Software Architecture § 18 (Lifecycle Management)
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

logger = logging.getLogger("BioOncoGraph")

# ── Thread-safe event ID generator ─────────────────────────────────

_id_lock = threading.Lock()
_id_counter = 0


def _next_event_id() -> str:
    """
    Generate a unique, monotonically increasing event identifier.

    Thread-safe via a module-level lock.

    Returns:
        A string of the form ``evt-00000001``.
    """
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return f"evt-{_id_counter:08d}"


# ── Priority ────────────────────────────────────────────────────────


class EventPriority(IntEnum):
    """
    Priority levels for event listeners.

    Lower numeric value means higher priority.  Listeners with
    ``HIGH`` priority are always dispatched before ``NORMAL``, which
    in turn run before ``LOW``.  Within the same priority, listeners
    execute in subscription (FIFO) order.
    """

    HIGH = 1
    NORMAL = 2
    LOW = 3


# ── Event base class ───────────────────────────────────────────────


@dataclass
class Event:
    """
    Base event carrying a payload through the EventBus.

    Attributes:
        source: Identifier of the component that emitted the event.
        name: Human-readable event name (overridden by subclasses).
        payload: Arbitrary key-value data attached to the event.
        id: Unique, monotonically increasing event identifier.
        timestamp: Unix timestamp at creation time.
    """

    source: str
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_next_event_id)
    timestamp: float = field(default_factory=time.time)


# ── Predefined lifecycle events ────────────────────────────────────


@dataclass
class ComponentInitializing(Event):
    """Emitted before a component's ``initialize()`` is called."""

    name: str = "component.initializing"


@dataclass
class ComponentInitialized(Event):
    """Emitted after a component's ``initialize()`` completes."""

    name: str = "component.initialized"


@dataclass
class ComponentStarting(Event):
    """Emitted before a component's ``start()`` is called."""

    name: str = "component.starting"


@dataclass
class ComponentStarted(Event):
    """Emitted after a component's ``start()`` completes."""

    name: str = "component.started"


@dataclass
class ComponentStopping(Event):
    """Emitted before a component's ``stop()`` is called."""

    name: str = "component.stopping"


@dataclass
class ComponentStopped(Event):
    """Emitted after a component's ``stop()`` completes."""

    name: str = "component.stopped"


@dataclass
class PipelineBuilding(Event):
    """Emitted when the pipeline build process begins."""

    name: str = "pipeline.building"


@dataclass
class PipelineBuilt(Event):
    """Emitted when the pipeline has been fully built."""

    name: str = "pipeline.built"


@dataclass
class PipelineShutdown(Event):
    """Emitted when the pipeline shutdown process completes."""

    name: str = "pipeline.shutdown"


@dataclass
class ErrorOccurred(Event):
    """Emitted when an error occurs during lifecycle management."""

    name: str = "error.occurred"


# ── Listener type alias ────────────────────────────────────────────

Listener = Callable[[Event], None]

# Internal subscription record: (priority_value, insertion_order, listener)
_Sub = Tuple[int, int, Listener]


# ── EventBus ────────────────────────────────────────────────────────


class EventBus:
    """
    Thread-safe publish/subscribe event bus with priority dispatch.

    Listeners are dispatched synchronously in deterministic order:
    first by priority (HIGH before NORMAL before LOW), then by
    subscription order (FIFO) within the same priority.

    If a listener raises an exception, the error is logged and
    dispatch continues to the remaining listeners (exception
    isolation).
    """

    def __init__(self) -> None:
        """Initialise an empty event bus."""
        self._subscriptions: Dict[
            Type[Event], List[_Sub]
        ] = defaultdict(list)
        self._lock = threading.Lock()
        self._order_counter = 0

    def subscribe(
        self,
        event_type: Type[Event],
        listener: Listener,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """
        Subscribe a listener to an event type.

        Duplicate subscriptions (same event type and same listener
        object) are silently ignored.

        Args:
            event_type: The event class to listen for.
            listener: A callable accepting a single ``Event`` arg.
            priority: Dispatch priority.

        Raises:
            TypeError: If *listener* is not callable or *event_type*
                is not an ``Event`` subclass.
        """
        if not callable(listener):
            raise TypeError(
                f"Listener must be callable, got "
                f"{type(listener).__name__}."
            )
        if not isinstance(event_type, type) or not issubclass(
            event_type, Event
        ):
            raise TypeError(
                f"event_type must be a subclass of Event, "
                f"got {event_type!r}."
            )

        with self._lock:
            subs = self._subscriptions[event_type]
            # Duplicate guard (identity check)
            for _, _, existing in subs:
                if existing is listener:
                    return
            self._order_counter += 1
            subs.append(
                (priority.value, self._order_counter, listener)
            )
            # Sort for deterministic dispatch order
            subs.sort(key=lambda s: (s[0], s[1]))

        logger.info(
            "Subscribed listener to %s (priority=%s)",
            event_type.__name__,
            priority.name,
        )

    def unsubscribe(
        self, event_type: Type[Event], listener: Listener
    ) -> None:
        """
        Remove a listener from an event type.

        Args:
            event_type: The event class the listener is registered for.
            listener: The listener to remove.

        Raises:
            ValueError: If the listener is not subscribed.
        """
        with self._lock:
            if event_type not in self._subscriptions:
                raise ValueError(
                    f"No listeners registered for "
                    f"{event_type.__name__}."
                )
            subs = self._subscriptions[event_type]
            new_subs = [
                (p, o, fn) for p, o, fn in subs if fn is not listener
            ]
            if len(new_subs) == len(subs):
                raise ValueError(
                    f"Listener not found for "
                    f"{event_type.__name__}."
                )
            if new_subs:
                self._subscriptions[event_type] = new_subs
            else:
                del self._subscriptions[event_type]

        logger.info(
            "Unsubscribed listener from %s",
            event_type.__name__,
        )

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed listeners.

        Listeners are called synchronously in priority then FIFO
        order.  If a listener raises, the exception is logged and
        dispatch continues to the next listener.

        Args:
            event: The event instance to publish.

        Raises:
            TypeError: If *event* is not an ``Event`` instance.
        """
        if not isinstance(event, Event):
            raise TypeError(
                f"Expected Event instance, got "
                f"{type(event).__name__}."
            )

        with self._lock:
            listeners = list(
                self._subscriptions.get(type(event), [])
            )

        for _, _, listener in listeners:
            try:
                listener(event)
            except Exception:
                logger.exception(
                    "Error in listener for event '%s' (id=%s)",
                    event.name,
                    event.id,
                )

        logger.debug(
            "Published: %s (id=%s, source=%s)",
            event.name,
            event.id,
            event.source,
        )

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            self._order_counter = 0
        logger.info("EventBus cleared")

    def listener_count(
        self, event_type: Optional[Type[Event]] = None
    ) -> int:
        """
        Count registered listeners.

        Args:
            event_type: If provided, count only listeners for this
                type.  If ``None``, count all listeners across all
                event types.

        Returns:
            The number of listeners.
        """
        with self._lock:
            if event_type is not None:
                return len(
                    self._subscriptions.get(event_type, [])
                )
            return sum(
                len(subs) for subs in self._subscriptions.values()
            )

    def registered_events(self) -> List[Type[Event]]:
        """
        List event types that have at least one listener.

        Returns:
            A list of event type classes.
        """
        with self._lock:
            return [
                et
                for et, subs in self._subscriptions.items()
                if subs
            ]

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"EventBus(events={len(self.registered_events())}, "
            f"listeners={self.listener_count()})"
        )
