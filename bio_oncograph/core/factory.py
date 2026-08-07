"""
Component Factory for Bio-OncoGraph.

Provides ``ComponentFactory``, which instantiates components by looking
up their class in a ``ComponentRegistry`` and introspecting the
constructor signature with ``inspect.signature``.  Extra config keys
that do not match constructor parameters are silently ignored; missing
mandatory parameters raise a clear ``TypeError``.

No ``eval()``, ``exec()``, or dynamic code execution is used.

Design references:
  - API Bible § 14 (Dependency Injection Contracts)
  - Software Architecture § 11 (Factory Pattern)
"""

import inspect
import logging
from typing import Any, Dict, List, Sequence, Type

from omegaconf import DictConfig, OmegaConf

from bio_oncograph.core.registry import ComponentRegistry

logger = logging.getLogger("BioOncoGraph")


class ComponentFactory:
    """
    Factory that creates component instances from a
    ``ComponentRegistry`` using constructor introspection.

    The factory:
      1. Looks up the class by name in the registry.
      2. Inspects the constructor signature via ``inspect.signature``.
      3. Filters kwargs to only those accepted by the constructor.
      4. Validates that all mandatory (no-default) parameters are
         provided.
      5. Instantiates and returns the object.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        """
        Initialize the factory with a backing registry.

        Args:
            registry: The ``ComponentRegistry`` to look up classes from.
        """
        self._registry = registry

    @property
    def registry(self) -> ComponentRegistry:
        """Return the backing registry."""
        return self._registry

    def create(self, name: str, **kwargs: Any) -> Any:
        """
        Create a component instance by name.

        Extra kwargs that do not match constructor parameters are
        silently filtered out.  Missing mandatory parameters raise
        ``TypeError``.

        Args:
            name: Registered component name.
            **kwargs: Constructor keyword arguments.

        Returns:
            The instantiated component.

        Raises:
            KeyError: If *name* is not in the registry.
            TypeError: If a mandatory constructor parameter is missing.
        """
        cls = self._registry.get(name)
        filtered = self._filter_kwargs(cls, kwargs)
        self._validate_required_args(cls, filtered)
        instance = cls(**filtered)
        param_str = ", ".join(
            f"{k}={v!r}" for k, v in filtered.items()
        )
        logger.info("Factory created: %s(%s)", name, param_str)
        return instance

    def create_from_config(self, cfg: DictConfig) -> Any:
        """
        Create a component from a Hydra ``DictConfig``.

        The config must contain a ``name`` key identifying the
        registered component.  All other keys are forwarded as
        constructor kwargs (after OmegaConf resolution).

        Args:
            cfg: A ``DictConfig`` with at least a ``name`` field.

        Returns:
            The instantiated component.

        Raises:
            ValueError: If *cfg* does not contain ``name``.
            KeyError: If the component is not registered.
            TypeError: If mandatory constructor args are missing.
        """
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError(
                "Config must resolve to a dictionary."
            )
        if "name" not in resolved:
            raise ValueError(
                "Config must contain a 'name' field identifying "
                "the component to create."
            )
        name = str(resolved.pop("name"))
        return self.create(name, **resolved)

    def create_many(
        self, configs: Sequence[DictConfig]
    ) -> List[Any]:
        """
        Create multiple components from a sequence of configs.

        Args:
            configs: A sequence of ``DictConfig`` objects, each
                     containing a ``name`` key.

        Returns:
            A list of instantiated components in the same order.
        """
        return [self.create_from_config(cfg) for cfg in configs]

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _filter_kwargs(
        cls: Type[Any], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter *kwargs* to only those accepted by *cls.__init__*.

        If the constructor accepts ``**kwargs`` (VAR_KEYWORD), all
        keys are passed through unfiltered.

        Args:
            cls: The target class.
            kwargs: Candidate keyword arguments.

        Returns:
            A filtered dictionary of valid kwargs.
        """
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            return dict(kwargs)

        return {k: v for k, v in kwargs.items() if k in valid_params}

    @staticmethod
    def _validate_required_args(
        cls: Type[Any], kwargs: Dict[str, Any]
    ) -> None:
        """
        Validate that all mandatory constructor params are present.

        Args:
            cls: The target class.
            kwargs: The kwargs that will be passed to the constructor.

        Raises:
            TypeError: If any mandatory parameter is missing.
        """
        sig = inspect.signature(cls.__init__)
        missing: List[str] = []
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if (
                param.default is inspect.Parameter.empty
                and param_name not in kwargs
            ):
                missing.append(param_name)

        if missing:
            raise TypeError(
                f"Missing required argument(s) {missing} for "
                f"{cls.__name__}. Provided: {list(kwargs.keys())}"
            )
