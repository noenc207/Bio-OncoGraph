"""
Callback framework for Bio-OncoGraph.

Provides:
  - ``CallbackPriority`` — dispatch priority enum.
  - ``Callback`` — base class with 13 no-op lifecycle hooks.
  - ``CallbackManager`` — manages, orders, and dispatches callbacks
    with exception isolation.

All types are pure Python — **no** ``torch``, ``tensorflow``,
``pytorch_lightning``, ``wandb``, or ``tensorboard`` imports.

Design references:
  - Software Architecture § 17 (Callback System)
  - API Bible § 12 (Callback Hooks)
"""

import logging
import threading
from enum import IntEnum
from typing import Any, Dict, List

logger = logging.getLogger("BioOncoGraph")


# ── Priority ────────────────────────────────────────────────────────


class CallbackPriority(IntEnum):
    """
    Dispatch priority for callbacks.

    Lower numeric value = higher priority.  Callbacks with equal
    priority are dispatched in insertion (registration) order.
    """

    HIGH = 0
    NORMAL = 50
    LOW = 100


# ── Callback Base ───────────────────────────────────────────────────


class Callback:
    """
    Base class for training callbacks.

    All hooks have no-op default implementations, so subclasses
    only need to override the hooks they care about.

    Attributes:
        priority: Dispatch priority (default ``NORMAL``).
    """

    priority: CallbackPriority = CallbackPriority.NORMAL

    # ── Training lifecycle ──────────────────────────────────────

    def on_fit_start(self, **kwargs: Any) -> None:
        """Called once before the first training epoch."""
        pass

    def on_fit_end(self, **kwargs: Any) -> None:
        """Called once after the last training epoch."""
        pass

    def on_epoch_start(self, **kwargs: Any) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, **kwargs: Any) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, **kwargs: Any) -> None:
        """Called before each training batch."""
        pass

    def on_batch_end(self, **kwargs: Any) -> None:
        """Called after each training batch."""
        pass

    # ── Validation lifecycle ────────────────────────────────────

    def on_validation_start(self, **kwargs: Any) -> None:
        """Called before a validation pass."""
        pass

    def on_validation_end(self, **kwargs: Any) -> None:
        """Called after a validation pass."""
        pass

    # ── Test lifecycle ──────────────────────────────────────────

    def on_test_start(self, **kwargs: Any) -> None:
        """Called before a test pass."""
        pass

    def on_test_end(self, **kwargs: Any) -> None:
        """Called after a test pass."""
        pass

    # ── Predict lifecycle ───────────────────────────────────────

    def on_predict_start(self, **kwargs: Any) -> None:
        """Called before a prediction pass."""
        pass

    def on_predict_end(self, **kwargs: Any) -> None:
        """Called after a prediction pass."""
        pass

    # ── Error handling ──────────────────────────────────────────

    def on_exception(self, **kwargs: Any) -> None:
        """Called when an unhandled exception is caught."""
        pass


# ── Valid hook names (for dispatch validation) ──────────────────────

_HOOK_NAMES = frozenset(
    name for name in dir(Callback)
    if name.startswith("on_") and callable(getattr(Callback, name))
)


# ── Callback Manager ───────────────────────────────────────────────


class CallbackManager:
    """
    Manages callback registration, ordering, and dispatch.

    Features:
      - Priority-ordered dispatch (HIGH → NORMAL → LOW).
      - Stable insertion-order within the same priority.
      - Exception isolation: a failing callback is logged and
        skipped, allowing subsequent callbacks to execute.
      - Thread-safe registration and dispatch.

    Example::

        mgr = CallbackManager()
        mgr.add(MyCallback())
        mgr.dispatch("on_epoch_end", epoch=5, metrics={...})
    """

    def __init__(self) -> None:
        """Initialise an empty manager."""
        self._callbacks: List[Callback] = []
        self._insertion_order: Dict[int, int] = {}
        self._next_order: int = 0
        self._lock = threading.Lock()

    def add(self, callback: Callback) -> None:
        """
        Register a callback.

        Args:
            callback: The callback instance.

        Raises:
            TypeError: If *callback* is not a ``Callback``.
            ValueError: If the exact same instance is already
                registered (identity check).
        """
        if not isinstance(callback, Callback):
            raise TypeError(
                f"Expected Callback, got "
                f"{type(callback).__name__}."
            )
        with self._lock:
            if any(cb is callback for cb in self._callbacks):
                raise ValueError(
                    f"Callback already registered: "
                    f"{type(callback).__name__}."
                )
            self._insertion_order[id(callback)] = self._next_order
            self._next_order += 1
            self._callbacks.append(callback)
            self._sort_locked()
        logger.debug(
            "Added callback: %s (priority=%s)",
            type(callback).__name__,
            callback.priority.name,
        )

    def remove(self, callback: Callback) -> None:
        """
        Remove a callback by identity.

        Args:
            callback: The exact instance to remove.

        Raises:
            ValueError: If the instance is not registered.
        """
        with self._lock:
            for i, cb in enumerate(self._callbacks):
                if cb is callback:
                    self._callbacks.pop(i)
                    self._insertion_order.pop(id(callback), None)
                    logger.debug(
                        "Removed callback: %s",
                        type(callback).__name__,
                    )
                    return
            raise ValueError(
                f"Callback not found: "
                f"{type(callback).__name__}."
            )

    def clear(self) -> None:
        """Remove all callbacks."""
        with self._lock:
            self._callbacks.clear()
            self._insertion_order.clear()
            self._next_order = 0
        logger.debug("Cleared all callbacks")

    def dispatch(self, event_name: str, **kwargs: Any) -> None:
        """
        Invoke a named hook on all registered callbacks.

        Callbacks are invoked in priority order (HIGH first).
        Exceptions are caught, logged, and do not prevent
        subsequent callbacks from executing.

        Args:
            event_name: The hook method name (e.g.
                ``"on_epoch_end"``).
            **kwargs: Arbitrary keyword arguments forwarded to
                each callback's hook.
        """
        with self._lock:
            snapshot = list(self._callbacks)

        for cb in snapshot:
            method = getattr(cb, event_name, None)
            if method is not None and callable(method):
                try:
                    method(**kwargs)
                except Exception:
                    logger.exception(
                        "Error in %s.%s",
                        type(cb).__name__,
                        event_name,
                    )

    def list_callbacks(self) -> List[str]:
        """
        Return callback class names in dispatch order.

        Returns:
            List of class-name strings.
        """
        with self._lock:
            return [type(cb).__name__ for cb in self._callbacks]

    def __len__(self) -> int:
        """Return the number of registered callbacks."""
        with self._lock:
            return len(self._callbacks)

    def __repr__(self) -> str:
        return (
            f"CallbackManager(callbacks="
            f"{self.list_callbacks()})"
        )

    # ── Private helpers ─────────────────────────────────────────

    def _sort_locked(self) -> None:
        """Sort callbacks by (priority, insertion_order). Lock held."""
        self._callbacks.sort(
            key=lambda cb: (
                cb.priority,
                self._insertion_order.get(id(cb), 0),
            )
        )

class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int = 10):
        self.patience = patience
    def on_epoch_end(self, **kwargs: Any) -> None: pass

class ModelCheckpointCallback(Callback):
    def __init__(self, save_best: bool = True, save_latest: bool = True):
        self.save_best = save_best
        self.save_latest = save_latest
    def on_epoch_end(self, **kwargs: Any) -> None: pass

class NanGuardCallback(Callback):
    def on_batch_end(self, **kwargs: Any) -> None: pass
