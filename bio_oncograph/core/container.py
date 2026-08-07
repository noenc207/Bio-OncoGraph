"""
Dependency Injection Container for Bio-OncoGraph.

Provides a thread-safe ``DIContainer`` that manages component
registration, automatic constructor injection, singleton/transient
lifetimes, and circular dependency detection.

Constructor injection works by matching parameter names in
``__init__`` signatures against registered component names.  No
``eval()`` or ``exec()`` is used; all introspection goes through
``inspect.signature``.

Design references:
  - Software Architecture § 10 (Dependency Injection)
  - API Bible § 14 (DI Contracts)
  - ADR-007 (Dependency Injection)
"""

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type

from bio_oncograph.core.exceptions import (
    CircularDependencyError,
    ContainerError,
    DependencyNotFoundError,
)

logger = logging.getLogger("BioOncoGraph")


@dataclass
class _Registration:
    """Internal record describing how to create a component."""

    cls: Optional[Type[Any]] = None
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    singleton: bool = True
    resolved: Optional[Any] = field(default=None, repr=False)


class DIContainer:
    """
    Thread-safe Dependency Injection Container.

    Supports three registration strategies:

    * **Class** (``register``): the container introspects the
      constructor, resolves parameters by name from the container,
      and instantiates the class.
    * **Instance** (``register_instance``): a pre-built object is
      stored directly (always singleton).
    * **Factory** (``register_factory``): a zero-argument callable
      is invoked to produce the instance.

    Singletons are created once and cached.  Transient components are
    re-created on every ``resolve`` call.

    Circular dependencies are detected per-thread via a resolution
    stack stored in ``threading.local``.
    """

    def __init__(self) -> None:
        """Initialise an empty container."""
        self._registrations: Dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._resolving: threading.local = threading.local()

    # ── Registration ────────────────────────────────────────────

    def register(
        self,
        name: str,
        cls: Type[Any],
        singleton: bool = True,
    ) -> None:
        """
        Register a class for constructor injection.

        Args:
            name: Unique component identifier.
            cls: The class to instantiate on ``resolve``.
            singleton: If ``True``, cache after first creation.

        Raises:
            ContainerError: If *name* is already registered.
            TypeError: If *cls* is not a type.
        """
        if not isinstance(cls, type):
            raise TypeError(
                f"Expected a class, got {type(cls).__name__} "
                f"for '{name}'."
            )
        with self._lock:
            self._assert_not_registered(name)
            self._registrations[name] = _Registration(
                cls=cls, singleton=singleton
            )
        logger.info("Registered: %s (singleton=%s)", name, singleton)

    def register_instance(self, name: str, instance: Any) -> None:
        """
        Register a pre-built instance (always singleton).

        Args:
            name: Unique component identifier.
            instance: The object to return on ``resolve``.

        Raises:
            ContainerError: If *name* is already registered.
        """
        with self._lock:
            self._assert_not_registered(name)
            reg = _Registration(
                instance=instance, singleton=True
            )
            reg.resolved = instance
            self._registrations[name] = reg
        logger.info("Registered instance: %s", name)

    def register_factory(
        self,
        name: str,
        factory: Callable[..., Any],
        singleton: bool = True,
    ) -> None:
        """
        Register a zero-argument factory callable.

        Args:
            name: Unique component identifier.
            factory: Callable that produces the component.
            singleton: If ``True``, call the factory only once.

        Raises:
            ContainerError: If *name* is already registered.
        """
        with self._lock:
            self._assert_not_registered(name)
            self._registrations[name] = _Registration(
                factory=factory, singleton=singleton
            )
        logger.info(
            "Registered factory: %s (singleton=%s)", name, singleton
        )

    # ── Resolution ──────────────────────────────────────────────

    def resolve(self, name: str) -> Any:
        """
        Resolve a component by name, injecting constructor deps.

        For singletons, the instance is cached after first creation.
        Circular dependencies are detected and raise immediately.

        Args:
            name: The component identifier.

        Returns:
            The resolved component instance.

        Raises:
            DependencyNotFoundError: If *name* is not registered.
            CircularDependencyError: If a cycle is detected.
        """
        resolving_stack = self._get_resolving_stack()

        if name in resolving_stack:
            chain = " → ".join(resolving_stack) + f" → {name}"
            raise CircularDependencyError(
                f"Circular dependency detected: {chain}"
            )

        with self._lock:
            if name not in self._registrations:
                raise DependencyNotFoundError(
                    f"Component '{name}' not found. "
                    f"Available: {self.list_components()}"
                )

            reg = self._registrations[name]

            # Fast path: cached singleton
            if reg.singleton and reg.resolved is not None:
                return reg.resolved

            resolving_stack.add(name)
            try:
                instance = self._create(name, reg)
            finally:
                resolving_stack.discard(name)

            if reg.singleton:
                reg.resolved = instance

            logger.info("Resolved: %s", name)
            return instance

    # ── Query ───────────────────────────────────────────────────

    def contains(self, name: str) -> bool:
        """
        Check whether a component is registered.

        Args:
            name: The component identifier.

        Returns:
            ``True`` if registered.
        """
        with self._lock:
            return name in self._registrations

    def list_components(self) -> List[str]:
        """
        List registered component names in insertion order.

        Returns:
            Ordered list of component name strings.
        """
        with self._lock:
            return list(self._registrations.keys())

    def get_dependencies(self, name: str) -> List[str]:
        """
        Return names of registered components that *name* depends on.

        Inspects the constructor signature and returns parameter names
        that match other registered components.  For factory and
        instance registrations, returns an empty list.

        Args:
            name: The component identifier.

        Returns:
            List of dependency names.

        Raises:
            DependencyNotFoundError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._registrations:
                raise DependencyNotFoundError(
                    f"Component '{name}' not found."
                )
            reg = self._registrations[name]
            if reg.cls is None:
                return []
            return self._inspect_deps(reg.cls)

    # ── Mutation ────────────────────────────────────────────────

    def remove(self, name: str) -> None:
        """
        Remove a registered component.

        Args:
            name: The component identifier.

        Raises:
            ContainerError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._registrations:
                raise ContainerError(
                    f"Component '{name}' not found."
                )
            del self._registrations[name]
        logger.info("Removed: %s", name)

    def clear(self) -> None:
        """Remove all registered components."""
        with self._lock:
            self._registrations.clear()
        logger.info("Container cleared")

    # ── Dunder ──────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of registered components."""
        with self._lock:
            return len(self._registrations)

    def __contains__(self, name: object) -> bool:
        """Support the ``in`` operator."""
        if not isinstance(name, str):
            return False
        return self.contains(name)

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"DIContainer(components={self.list_components()})"
        )

    # ── Private helpers ─────────────────────────────────────────

    def _get_resolving_stack(self) -> Set[str]:
        """Return the per-thread resolution stack."""
        if not hasattr(self._resolving, "stack"):
            self._resolving.stack: Set[str] = set()
        return self._resolving.stack  # type: ignore[return-value]

    def _assert_not_registered(self, name: str) -> None:
        """Raise if *name* is already registered."""
        if name in self._registrations:
            raise ContainerError(
                f"Component '{name}' is already registered."
            )

    def _create(self, name: str, reg: _Registration) -> Any:
        """Instantiate a component from its registration."""
        if reg.instance is not None:
            return reg.instance
        if reg.factory is not None:
            return reg.factory()
        if reg.cls is not None:
            kwargs = self._inject_constructor(reg.cls)
            return reg.cls(**kwargs)
        raise ContainerError(
            f"Registration for '{name}' has no class, factory, "
            f"or instance."
        )

    def _inject_constructor(
        self, cls: Type[Any]
    ) -> Dict[str, Any]:
        """Resolve constructor parameters from the container."""
        sig = inspect.signature(cls.__init__)
        kwargs: Dict[str, Any] = {}

        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if pname in self._registrations:
                kwargs[pname] = self.resolve(pname)
            elif param.default is not inspect.Parameter.empty:
                pass  # Use constructor default
            else:
                raise DependencyNotFoundError(
                    f"Cannot resolve parameter '{pname}' for "
                    f"{cls.__name__}. Not registered and has "
                    f"no default value."
                )
        return kwargs

    def _inspect_deps(self, cls: Type[Any]) -> List[str]:
        """Return registered names that appear as ctor params."""
        sig = inspect.signature(cls.__init__)
        deps: List[str] = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if pname in self._registrations:
                deps.append(pname)
        return deps
