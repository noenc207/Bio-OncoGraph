"""
Unit tests for Checkpoint and CheckpointManager.

Covers: Checkpoint ABC, InMemoryCheckpoint stub (save/load/latest/
exists/list_checkpoints), CheckpointManager (register_backend,
save/load/latest, default backend, duplicate backend, missing
backend, load nonexistent, multiple backends).
"""

from typing import Any, Dict, List, Optional

import pytest

from bio_oncograph.core.checkpoint import (
    Checkpoint,
    CheckpointManager,
)


# ── In-memory stub (for testing only) ──────────────────────────────


class InMemoryCheckpoint(Checkpoint):
    """In-memory checkpoint backend for unit testing."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def save(self, state: Dict[str, Any], identifier: str) -> None:
        self._store[identifier] = dict(state)
        if identifier not in self._order:
            self._order.append(identifier)

    def load(self, identifier: str) -> Dict[str, Any]:
        if identifier not in self._store:
            raise KeyError(
                f"Checkpoint '{identifier}' not found."
            )
        return dict(self._store[identifier])

    def latest(self) -> Optional[str]:
        if not self._order:
            return None
        return self._order[-1]

    def exists(self, identifier: str) -> bool:
        return identifier in self._store

    def list_checkpoints(self) -> List[str]:
        return list(self._order)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def backend() -> InMemoryCheckpoint:
    return InMemoryCheckpoint()


@pytest.fixture()
def manager() -> CheckpointManager:
    return CheckpointManager()


# ══════════════════════════════════════════════════════════════════
# Checkpoint ABC
# ══════════════════════════════════════════════════════════════════


def test_checkpoint_abc_cannot_instantiate() -> None:
    """Checkpoint is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        Checkpoint()  # type: ignore[abstract]


# ══════════════════════════════════════════════════════════════════
# InMemoryCheckpoint (backend contract)
# ══════════════════════════════════════════════════════════════════


def test_save_and_load(backend: InMemoryCheckpoint) -> None:
    """save/load round-trips a state dict."""
    state = {"epoch": 5, "loss": 0.3}
    backend.save(state, "ckpt_05")
    loaded = backend.load("ckpt_05")
    assert loaded == state


def test_load_returns_copy(backend: InMemoryCheckpoint) -> None:
    """load returns a copy, not a reference."""
    state = {"epoch": 1}
    backend.save(state, "ckpt")
    loaded = backend.load("ckpt")
    loaded["epoch"] = 999
    assert backend.load("ckpt")["epoch"] == 1


def test_load_nonexistent_raises(
    backend: InMemoryCheckpoint,
) -> None:
    """load raises KeyError for unknown identifier."""
    with pytest.raises(KeyError):
        backend.load("nonexistent")


def test_latest_empty(backend: InMemoryCheckpoint) -> None:
    """latest returns None when no checkpoints."""
    assert backend.latest() is None


def test_latest_returns_last(
    backend: InMemoryCheckpoint,
) -> None:
    """latest returns the most recently saved identifier."""
    backend.save({"a": 1}, "first")
    backend.save({"b": 2}, "second")
    assert backend.latest() == "second"


def test_exists_true(backend: InMemoryCheckpoint) -> None:
    """exists returns True for saved checkpoint."""
    backend.save({}, "x")
    assert backend.exists("x") is True


def test_exists_false(backend: InMemoryCheckpoint) -> None:
    """exists returns False for unknown identifier."""
    assert backend.exists("unknown") is False


def test_list_checkpoints(
    backend: InMemoryCheckpoint,
) -> None:
    """list_checkpoints returns identifiers in order."""
    backend.save({}, "a")
    backend.save({}, "b")
    backend.save({}, "c")
    assert backend.list_checkpoints() == ["a", "b", "c"]


def test_list_checkpoints_empty(
    backend: InMemoryCheckpoint,
) -> None:
    """list_checkpoints returns empty list initially."""
    assert backend.list_checkpoints() == []


# ══════════════════════════════════════════════════════════════════
# CheckpointManager — register_backend
# ══════════════════════════════════════════════════════════════════


def test_register_backend(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """register_backend adds a backend."""
    manager.register_backend("mem", backend)
    assert "mem" in manager.list_backends()


def test_register_backend_default(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """First registered backend becomes default."""
    manager.register_backend("mem", backend)
    assert manager.default_backend == "mem"


def test_register_backend_explicit_default(
    manager: CheckpointManager,
) -> None:
    """default=True overrides the current default."""
    manager.register_backend("a", InMemoryCheckpoint())
    manager.register_backend(
        "b", InMemoryCheckpoint(), default=True
    )
    assert manager.default_backend == "b"


def test_register_duplicate_raises(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """Registering the same name twice raises ValueError."""
    manager.register_backend("dup", backend)
    with pytest.raises(ValueError, match="already registered"):
        manager.register_backend("dup", InMemoryCheckpoint())


def test_register_non_checkpoint_raises(
    manager: CheckpointManager,
) -> None:
    """Registering a non-Checkpoint raises TypeError."""
    with pytest.raises(TypeError, match="Expected Checkpoint"):
        manager.register_backend(
            "bad", "not_a_backend"  # type: ignore[arg-type]
        )


# ══════════════════════════════════════════════════════════════════
# CheckpointManager — save / load / latest
# ══════════════════════════════════════════════════════════════════


def test_save_and_load_via_manager(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """save/load through manager round-trips correctly."""
    manager.register_backend("mem", backend)
    manager.save_checkpoint({"epoch": 10}, "ckpt_10")
    loaded = manager.load_checkpoint("ckpt_10")
    assert loaded["epoch"] == 10


def test_latest_via_manager(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """latest_checkpoint returns the latest identifier."""
    manager.register_backend("mem", backend)
    manager.save_checkpoint({}, "a")
    manager.save_checkpoint({}, "b")
    assert manager.latest_checkpoint() == "b"


def test_load_nonexistent_via_manager(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """load nonexistent checkpoint raises KeyError."""
    manager.register_backend("mem", backend)
    with pytest.raises(KeyError, match="not found"):
        manager.load_checkpoint("missing")


def test_no_backend_raises(
    manager: CheckpointManager,
) -> None:
    """Operations without backend raise ValueError."""
    with pytest.raises(ValueError, match="No checkpoint backend"):
        manager.save_checkpoint({}, "x")


def test_unknown_backend_raises(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """Specifying unknown backend raises KeyError."""
    manager.register_backend("mem", backend)
    with pytest.raises(KeyError, match="not registered"):
        manager.save_checkpoint({}, "x", backend="cloud")


# ══════════════════════════════════════════════════════════════════
# CheckpointManager — multi-backend
# ══════════════════════════════════════════════════════════════════


def test_multi_backend_routing(
    manager: CheckpointManager,
) -> None:
    """Explicit backend routes to the correct store."""
    b1 = InMemoryCheckpoint()
    b2 = InMemoryCheckpoint()
    manager.register_backend("local", b1, default=True)
    manager.register_backend("cloud", b2)

    manager.save_checkpoint({"src": "local"}, "c1")
    manager.save_checkpoint(
        {"src": "cloud"}, "c2", backend="cloud"
    )

    assert manager.load_checkpoint("c1")["src"] == "local"
    assert manager.load_checkpoint(
        "c2", backend="cloud"
    )["src"] == "cloud"

    # c2 not in local
    with pytest.raises(KeyError):
        manager.load_checkpoint("c2")  # default=local


def test_latest_per_backend(
    manager: CheckpointManager,
) -> None:
    """latest_checkpoint respects the backend parameter."""
    b1 = InMemoryCheckpoint()
    b2 = InMemoryCheckpoint()
    manager.register_backend("a", b1, default=True)
    manager.register_backend("b", b2)

    manager.save_checkpoint({}, "x1")
    manager.save_checkpoint({}, "y1", backend="b")

    assert manager.latest_checkpoint() == "x1"
    assert manager.latest_checkpoint(backend="b") == "y1"


def test_list_backends(
    manager: CheckpointManager,
) -> None:
    """list_backends returns registered names."""
    manager.register_backend("a", InMemoryCheckpoint())
    manager.register_backend("b", InMemoryCheckpoint())
    assert manager.list_backends() == ["a", "b"]


def test_repr(
    manager: CheckpointManager, backend: InMemoryCheckpoint
) -> None:
    """repr includes backend names."""
    manager.register_backend("mem", backend)
    assert "mem" in repr(manager)
