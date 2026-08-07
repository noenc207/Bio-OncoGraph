"""
Component Registry for Bio-OncoGraph.

Provides a thread-safe, generic ``ComponentRegistry`` class that maps
string names to Python classes.  Used by ``ComponentFactory`` to
instantiate modules from Hydra configuration without hard-coding any
concrete class.

Backward-compatible module-level functions (``register_module``,
``build_from_config``, ``load_plugins``) are preserved as thin wrappers
around ``ComponentRegistry`` instances grouped by functional category.

Design references:
  - API Bible § 16 (Registry API)
  - Software Architecture § 10–12 (DI, Factory, Registry)
  - ADR-007 (Dependency Injection)
"""

import importlib
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Type

from omegaconf import DictConfig

from bio_oncograph.core.exceptions import ConfigException

logger = logging.getLogger("BioOncoGraph")


class ComponentRegistry:
    """
    Thread-safe, deterministically-ordered component registry.

    Each registry instance manages a single functional group (e.g.
    ``"model"``, ``"loss"``, ``"pooling"``).  Components are stored in
    insertion order (Python 3.7+ ``dict`` guarantee).

    Attributes:
        name: Human-readable name for this registry group.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a new registry.

        Args:
            name: Identifier for this registry group (e.g. ``"model"``).
        """
        self._name = name
        self._components: Dict[str, Type[Any]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return the registry group name."""
        return self._name

    def register(self, name: str, cls: Type[Any]) -> None:
        """
        Register a component class under the given name.

        Args:
            name: Unique string identifier for the component.
            cls: The class to register.

        Raises:
            ValueError: If *name* is already registered.
            TypeError: If *cls* is not a type.
        """
        if not isinstance(cls, type):
            raise TypeError(
                f"Expected a class, got {type(cls).__name__} for '{name}'."
            )
        with self._lock:
            if name in self._components:
                raise ValueError(
                    f"Component '{name}' is already registered "
                    f"in registry '{self._name}'."
                )
            self._components[name] = cls
        logger.info("Registered component: %s in %s", name, self._name)

    def unregister(self, name: str) -> None:
        """
        Remove a previously registered component.

        Args:
            name: The identifier of the component to remove.

        Raises:
            KeyError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._components:
                raise KeyError(
                    f"Component '{name}' not found in "
                    f"registry '{self._name}'."
                )
            del self._components[name]
        logger.info("Unregistered component: %s from %s", name, self._name)

    def contains(self, name: str) -> bool:
        """
        Check whether a component is registered.

        Args:
            name: The identifier to look up.

        Returns:
            ``True`` if the component exists, ``False`` otherwise.
        """
        with self._lock:
            return name in self._components

    def get(self, name: str) -> Type[Any]:
        """
        Retrieve a registered component class by name.

        Args:
            name: The identifier of the component.

        Returns:
            The registered class.

        Raises:
            KeyError: If *name* is not registered.
        """
        with self._lock:
            if name not in self._components:
                available = list(self._components.keys())
                raise KeyError(
                    f"Component '{name}' not found in "
                    f"registry '{self._name}'. "
                    f"Available: {available}"
                )
            return self._components[name]

    def list_components(self) -> List[str]:
        """
        List all registered component names in insertion order.

        Returns:
            A list of component name strings.
        """
        with self._lock:
            return list(self._components.keys())

    def clear(self) -> None:
        """Remove all registered components."""
        with self._lock:
            self._components.clear()
        logger.info("Cleared all components from %s", self._name)

    def __len__(self) -> int:
        """Return the number of registered components."""
        with self._lock:
            return len(self._components)

    def __contains__(self, name: object) -> bool:
        """Support the ``in`` operator."""
        if not isinstance(name, str):
            return False
        return self.contains(name)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"ComponentRegistry(name={self._name!r}, "
            f"components={self.list_components()})"
        )


# ── Module-level grouped registries (backward compatibility) ────────

_group_registries: Dict[str, ComponentRegistry] = {}
_group_lock = threading.Lock()


def _get_or_create_group(group: str) -> ComponentRegistry:
    """
    Get or lazily create a ``ComponentRegistry`` for the given group.

    Args:
        group: Functional group name (e.g. ``"model"``).

    Returns:
        The ``ComponentRegistry`` instance for *group*.
    """
    with _group_lock:
        if group not in _group_registries:
            _group_registries[group] = ComponentRegistry(name=group)
        return _group_registries[group]


def register_module(
    name: str, group: str
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a class into a named group registry.

    This is the backward-compatible wrapper around
    ``ComponentRegistry.register``.

    Args:
        name: Unique identifier within the group.
        group: Functional group (e.g. ``"model"``, ``"loss"``).

    Returns:
        A class decorator.

    Raises:
        ConfigException: If *name* is already registered in *group*.
    """

    def wrapper(cls: Type[Any]) -> Type[Any]:
        registry = _get_or_create_group(group)
        try:
            registry.register(name, cls)
        except (ValueError, TypeError) as exc:
            raise ConfigException(str(exc)) from exc
        return cls

    return wrapper


def build_from_config(config: DictConfig, group: str) -> Any:
    """
    Instantiate a component from a Hydra config and a group registry.

    The config must contain a ``_target_name_`` key identifying the
    registered component.  Keys starting with ``_`` are stripped before
    being passed as constructor kwargs.

    This is the backward-compatible wrapper; prefer
    ``ComponentFactory.create_from_config`` for new code.

    Args:
        config: Hydra ``DictConfig`` with ``_target_name_`` and params.
        group: The registry group to look up.

    Returns:
        The instantiated component.

    Raises:
        ConfigException: If the target is missing or instantiation fails.
    """
    # Import here to avoid circular dependency at module level
    from bio_oncograph.core.factory import ComponentFactory

    if "_target_name_" not in config:
        raise ConfigException(
            "Config is missing the required '_target_name_' key."
        )

    target_name = str(config._target_name_)
    registry = _get_or_create_group(group)

    if not registry.contains(target_name):
        raise ConfigException(
            f"Target '{target_name}' not found in "
            f"registry group '{group}'."
        )

    factory = ComponentFactory(registry)
    params = {
        k: v for k, v in config.items() if not k.startswith("_")
    }

    try:
        return factory.create(target_name, **params)
    except (TypeError, KeyError) as exc:
        raise ConfigException(
            f"Failed to instantiate {target_name}: {exc}"
        ) from exc


def load_plugins() -> None:
    """
    Load external plugin modules from the ``BIOONCO_PLUGINS`` env-var.

    Format: ``"my_package.module1,my_package.module2"``

    Each module is imported, which triggers any ``@register_module``
    decorators it contains.

    Raises:
        ConfigException: If a plugin module cannot be imported.
    """
    plugins_env = os.environ.get("BIOONCO_PLUGINS", "")
    if not plugins_env:
        return

    for plugin_path in plugins_env.split(","):
        plugin_path = plugin_path.strip()
        if plugin_path:
            try:
                importlib.import_module(plugin_path)
            except ImportError as exc:
                raise ConfigException(
                    f"Failed to load plugin '{plugin_path}': {exc}"
                ) from exc
