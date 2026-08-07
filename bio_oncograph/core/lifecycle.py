"""
Lifecycle Manager for Bio-OncoGraph.

Provides ``LifecycleManager`` which orchestrates component lifecycle
transitions (initialise → start → stop → shutdown) and publishes
the corresponding lifecycle events through an ``EventBus``.

If a lifecycle method on a component raises an exception, an
``ErrorOccurred`` event is emitted before the exception is re-raised,
ensuring that observers are always notified of failures.

Design references:
  - Software Architecture § 18 (Lifecycle Management)
  - Software Architecture § 17 (Callback System)
"""

import logging
from typing import Any, List

from bio_oncograph.core.events import (
    ComponentInitialized,
    ComponentInitializing,
    ComponentStarted,
    ComponentStarting,
    ComponentStopped,
    ComponentStopping,
    ErrorOccurred,
    EventBus,
)

logger = logging.getLogger("BioOncoGraph")


class LifecycleManager:
    """
    Manages component lifecycle transitions and emits events.

    Each transition follows the pattern:

    1. Publish a *pre* event (e.g. ``ComponentInitializing``).
    2. Call the component's lifecycle method if it exists.
    3. Publish a *post* event (e.g. ``ComponentInitialized``).

    If step 2 raises, an ``ErrorOccurred`` event is published and
    the exception is re-raised to the caller.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialise the lifecycle manager.

        Args:
            event_bus: The event bus for publishing lifecycle events.
        """
        self._event_bus = event_bus
        self._initialized: List[str] = []
        self._started: List[str] = []

    @property
    def event_bus(self) -> EventBus:
        """Return the backing event bus."""
        return self._event_bus

    @property
    def initialized_components(self) -> List[str]:
        """Return names of currently initialised components."""
        return list(self._initialized)

    @property
    def started_components(self) -> List[str]:
        """Return names of currently started components."""
        return list(self._started)

    def initialize(
        self, component: Any, name: str = ""
    ) -> None:
        """
        Initialise a component.

        Publishes ``ComponentInitializing`` before and
        ``ComponentInitialized`` after calling
        ``component.initialize()``.

        Args:
            component: The component to initialise.
            name: Optional human-readable name.  Defaults to the
                class name.

        Raises:
            Exception: Re-raised from ``component.initialize()``.
        """
        comp_name = name or type(component).__name__
        self._event_bus.publish(
            ComponentInitializing(source=comp_name)
        )
        try:
            if hasattr(component, "initialize") and callable(
                getattr(component, "initialize")
            ):
                component.initialize()
        except Exception as exc:
            self._event_bus.publish(
                ErrorOccurred(
                    source=comp_name,
                    payload={
                        "error": str(exc),
                        "phase": "initialize",
                    },
                )
            )
            raise
        self._initialized.append(comp_name)
        self._event_bus.publish(
            ComponentInitialized(source=comp_name)
        )
        logger.info("Component initialized: %s", comp_name)

    def start(self, component: Any, name: str = "") -> None:
        """
        Start a component.

        Publishes ``ComponentStarting`` before and
        ``ComponentStarted`` after calling ``component.start()``.

        Args:
            component: The component to start.
            name: Optional human-readable name.

        Raises:
            Exception: Re-raised from ``component.start()``.
        """
        comp_name = name or type(component).__name__
        self._event_bus.publish(
            ComponentStarting(source=comp_name)
        )
        try:
            if hasattr(component, "start") and callable(
                getattr(component, "start")
            ):
                component.start()
        except Exception as exc:
            self._event_bus.publish(
                ErrorOccurred(
                    source=comp_name,
                    payload={
                        "error": str(exc),
                        "phase": "start",
                    },
                )
            )
            raise
        self._started.append(comp_name)
        self._event_bus.publish(
            ComponentStarted(source=comp_name)
        )
        logger.info("Component started: %s", comp_name)

    def stop(self, component: Any, name: str = "") -> None:
        """
        Stop a running component.

        Publishes ``ComponentStopping`` before and
        ``ComponentStopped`` after calling ``component.stop()``.

        Args:
            component: The component to stop.
            name: Optional human-readable name.

        Raises:
            Exception: Re-raised from ``component.stop()``.
        """
        comp_name = name or type(component).__name__
        self._event_bus.publish(
            ComponentStopping(source=comp_name)
        )
        try:
            if hasattr(component, "stop") and callable(
                getattr(component, "stop")
            ):
                component.stop()
        except Exception as exc:
            self._event_bus.publish(
                ErrorOccurred(
                    source=comp_name,
                    payload={
                        "error": str(exc),
                        "phase": "stop",
                    },
                )
            )
            raise
        self._started = [
            n for n in self._started if n != comp_name
        ]
        self._event_bus.publish(
            ComponentStopped(source=comp_name)
        )
        logger.info("Component stopped: %s", comp_name)

    def shutdown(self, component: Any, name: str = "") -> None:
        """
        Shut down a component (stop if running, then tear down).

        If the component is currently in the *started* list, ``stop``
        is called first.  Then ``component.shutdown()`` is invoked.
        Finally the component is removed from the *initialized* list.

        Args:
            component: The component to shut down.
            name: Optional human-readable name.

        Raises:
            Exception: Re-raised from ``component.shutdown()``.
        """
        comp_name = name or type(component).__name__

        # Auto-stop if currently started
        if comp_name in self._started:
            self.stop(component, name)

        self._event_bus.publish(
            ComponentStopping(source=comp_name)
        )
        try:
            if hasattr(component, "shutdown") and callable(
                getattr(component, "shutdown")
            ):
                component.shutdown()
        except Exception as exc:
            self._event_bus.publish(
                ErrorOccurred(
                    source=comp_name,
                    payload={
                        "error": str(exc),
                        "phase": "shutdown",
                    },
                )
            )
            raise
        self._initialized = [
            n for n in self._initialized if n != comp_name
        ]
        self._event_bus.publish(
            ComponentStopped(source=comp_name)
        )
        logger.info("Component shut down: %s", comp_name)
