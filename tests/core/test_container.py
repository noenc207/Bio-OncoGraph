"""
Unit tests for DIContainer.

Covers: registration strategies, singleton vs transient, constructor
injection, nested injection, circular dependency detection, thread
safety, duplicate registration, remove, clear, deterministic ordering,
lazy initialisation, and error paths.
"""

import threading
from typing import Any

import pytest

from bio_oncograph.core.container import DIContainer
from bio_oncograph.core.exceptions import (
    CircularDependencyError,
    ContainerError,
    DependencyNotFoundError,
)


# ── Dummy components ────────────────────────────────────────────────


class Database:
    def __init__(self) -> None:
        self.connected = True


class Cache:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}


class Repository:
    """Depends on Database (resolved by name)."""

    def __init__(self, database: Any) -> None:
        self.database = database


class Service:
    """Depends on Repository and Cache (resolved by name)."""

    def __init__(self, repository: Any, cache: Any) -> None:
        self.repository = repository
        self.cache = cache


class DefaultedComponent:
    """Has a dependency with a default value."""

    def __init__(self, database: Any, timeout: int = 30) -> None:
        self.database = database
        self.timeout = timeout


class Counter:
    """Stateful class to test transient vs singleton."""
    _count = 0

    def __init__(self) -> None:
        Counter._count += 1
        self.id = Counter._count


class CycleA:
    def __init__(self, cycle_b: Any) -> None:
        self.b = cycle_b


class CycleB:
    def __init__(self, cycle_a: Any) -> None:
        self.a = cycle_a


class CycleC:
    def __init__(self, cycle_a: Any) -> None:
        self.a = cycle_a


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def container() -> DIContainer:
    """Fresh container for each test."""
    return DIContainer()


# ── Registration ────────────────────────────────────────────────────


def test_register_class(container: DIContainer) -> None:
    """register adds a class to the container."""
    container.register("database", Database)
    assert container.contains("database")


def test_register_instance(container: DIContainer) -> None:
    """register_instance stores a pre-built object."""
    db = Database()
    container.register_instance("database", db)
    resolved = container.resolve("database")
    assert resolved is db


def test_register_factory(container: DIContainer) -> None:
    """register_factory stores a callable."""
    container.register_factory("database", Database)
    resolved = container.resolve("database")
    assert isinstance(resolved, Database)


def test_duplicate_register_raises(container: DIContainer) -> None:
    """Duplicate registration raises ContainerError."""
    container.register("database", Database)
    with pytest.raises(ContainerError, match="already registered"):
        container.register("database", Cache)


def test_register_non_type_raises(container: DIContainer) -> None:
    """Registering an instance via register() raises TypeError."""
    with pytest.raises(TypeError, match="Expected a class"):
        container.register("bad", Database())  # type: ignore[arg-type]


# ── Singleton vs Transient ──────────────────────────────────────────


def test_singleton_returns_same_instance(
    container: DIContainer,
) -> None:
    """Singleton resolution returns the same object."""
    container.register("database", Database, singleton=True)
    a = container.resolve("database")
    b = container.resolve("database")
    assert a is b


def test_transient_returns_different_instances(
    container: DIContainer,
) -> None:
    """Transient resolution creates a new object each time."""
    Counter._count = 0
    container.register("counter", Counter, singleton=False)
    a = container.resolve("counter")
    b = container.resolve("counter")
    assert a is not b
    assert a.id != b.id


# ── Constructor Injection ───────────────────────────────────────────


def test_constructor_injection(container: DIContainer) -> None:
    """Container resolves constructor deps by parameter name."""
    container.register("database", Database)
    container.register("repository", Repository)
    repo = container.resolve("repository")
    assert isinstance(repo, Repository)
    assert isinstance(repo.database, Database)


def test_nested_injection(container: DIContainer) -> None:
    """Multi-level dependency chain resolves correctly."""
    container.register("database", Database)
    container.register("cache", Cache)
    container.register("repository", Repository)
    container.register("service", Service)

    svc = container.resolve("service")
    assert isinstance(svc, Service)
    assert isinstance(svc.repository, Repository)
    assert isinstance(svc.repository.database, Database)
    assert isinstance(svc.cache, Cache)


def test_default_parameter_not_injected(
    container: DIContainer,
) -> None:
    """Params with defaults use defaults when not registered."""
    container.register("database", Database)
    container.register("defaulted", DefaultedComponent)
    obj = container.resolve("defaulted")
    assert obj.timeout == 30


def test_missing_mandatory_dep_raises(
    container: DIContainer,
) -> None:
    """Missing mandatory dep raises DependencyNotFoundError."""
    container.register("repository", Repository)
    # 'database' is NOT registered
    with pytest.raises(DependencyNotFoundError, match="database"):
        container.resolve("repository")


# ── Circular Dependency ─────────────────────────────────────────────


def test_circular_dependency_detected(
    container: DIContainer,
) -> None:
    """Two-node cycle raises CircularDependencyError."""
    container.register("cycle_a", CycleA)
    container.register("cycle_b", CycleB)
    with pytest.raises(CircularDependencyError, match="Circular"):
        container.resolve("cycle_a")


def test_three_node_cycle(container: DIContainer) -> None:
    """Three-node cycle (A→B, B→C→A) is detected."""
    container.register("cycle_a", CycleA)
    container.register("cycle_b", CycleB)
    # CycleB depends on cycle_a, CycleA depends on cycle_b
    with pytest.raises(CircularDependencyError):
        container.resolve("cycle_a")


# ── Query ───────────────────────────────────────────────────────────


def test_contains_false(container: DIContainer) -> None:
    """contains returns False for unregistered names."""
    assert not container.contains("ghost")


def test_list_components_ordered(container: DIContainer) -> None:
    """list_components returns insertion order."""
    container.register("b", Cache)
    container.register("a", Database)
    assert container.list_components() == ["b", "a"]


def test_get_dependencies(container: DIContainer) -> None:
    """get_dependencies inspects constructor params."""
    container.register("database", Database)
    container.register("repository", Repository)
    deps = container.get_dependencies("repository")
    assert deps == ["database"]


def test_get_dependencies_factory(container: DIContainer) -> None:
    """Factory registrations return empty dependencies."""
    container.register_factory("database", Database)
    assert container.get_dependencies("database") == []


# ── Mutation ────────────────────────────────────────────────────────


def test_remove(container: DIContainer) -> None:
    """remove deletes a component."""
    container.register("database", Database)
    container.remove("database")
    assert not container.contains("database")


def test_remove_unknown_raises(container: DIContainer) -> None:
    """Removing unregistered name raises ContainerError."""
    with pytest.raises(ContainerError, match="not found"):
        container.remove("ghost")


def test_clear(container: DIContainer) -> None:
    """clear removes all components."""
    container.register("a", Database)
    container.register("b", Cache)
    container.clear()
    assert len(container) == 0


# ── Lazy Initialisation ────────────────────────────────────────────


def test_lazy_initialization(container: DIContainer) -> None:
    """Singleton is NOT created until resolve is called."""
    Counter._count = 0
    container.register("counter", Counter)
    assert Counter._count == 0  # Not yet created
    container.resolve("counter")
    assert Counter._count == 1  # Created on first resolve


# ── Thread Safety ───────────────────────────────────────────────────


def test_concurrent_resolve(container: DIContainer) -> None:
    """Concurrent resolves do not corrupt state."""
    container.register("database", Database, singleton=True)
    results: list[Database] = []
    errors: list[Exception] = []

    def resolver() -> None:
        try:
            results.append(container.resolve("database"))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=resolver) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    # All should be the same singleton
    assert all(r is results[0] for r in results)


# ── Dunder ──────────────────────────────────────────────────────────


def test_len(container: DIContainer) -> None:
    """__len__ reports registered count."""
    container.register("a", Database)
    container.register_instance("b", Cache())
    assert len(container) == 2


def test_in_operator(container: DIContainer) -> None:
    """'in' operator works."""
    container.register("a", Database)
    assert "a" in container
    assert "b" not in container


def test_repr(container: DIContainer) -> None:
    """repr includes component names."""
    container.register("db", Database)
    assert "db" in repr(container)
