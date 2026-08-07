"""
Pipeline Builder for Bio-OncoGraph.

Provides ``PipelineBuilder`` which resolves a complete object graph
from a ``DIContainer``, manages component lifecycle (``initialize`` /
``shutdown``), and validates the dependency graph for cycles and
missing components using ``graphlib.TopologicalSorter``.

Design references:
  - Software Architecture § 18 (Lifecycle Management)
  - Software Architecture § 13 (Builder Pattern)
"""

import inspect
import logging
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any, Dict, List, Optional, Sequence, Set

from omegaconf import DictConfig

from bio_oncograph.core.container import DIContainer
from bio_oncograph.core.exceptions import (
    CircularDependencyError,
    DependencyNotFoundError,
    PipelineValidationError,
)

logger = logging.getLogger("BioOncoGraph")


@dataclass
class Pipeline:
    """
    Container holding fully resolved and initialised components.

    Attributes:
        components: Mapping from component name to resolved instance.
        build_order: Topologically sorted creation order.
    """

    components: Dict[str, Any] = field(default_factory=dict)
    build_order: List[str] = field(default_factory=list)

    def get(self, name: str) -> Any:
        """
        Retrieve a component by name.

        Args:
            name: The component identifier.

        Returns:
            The resolved component instance.

        Raises:
            KeyError: If *name* is not in the pipeline.
        """
        if name not in self.components:
            raise KeyError(
                f"Component '{name}' not in pipeline. "
                f"Available: {list(self.components.keys())}"
            )
        return self.components[name]

    def __contains__(self, name: object) -> bool:
        """Support the ``in`` operator."""
        if not isinstance(name, str):
            return False
        return name in self.components

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self.components)


class PipelineBuilder:
    """
    Builds and manages component pipelines from a ``DIContainer``.

    Workflow:
      1. ``validate`` — check for missing deps and cycles.
      2. ``build`` — resolve all components in topological order,
         then call ``initialize()`` on each (if present).
      3. ``shutdown`` — call ``shutdown()`` in reverse order.
      4. ``visualize`` — produce a textual dependency graph.
    """

    def __init__(self, container: DIContainer) -> None:
        """
        Initialise the builder with a container.

        Args:
            container: The ``DIContainer`` holding registrations.
        """
        self._container = container
        self._pipeline: Optional[Pipeline] = None
        self._build_order: List[str] = []

    @property
    def container(self) -> DIContainer:
        """Return the backing container."""
        return self._container

    # ── Public API ──────────────────────────────────────────────

    def build(self, config: DictConfig) -> Pipeline:
        """
        Build a pipeline from configuration.

        The config should contain a ``components`` key with a list
        of component names to resolve, or its top-level keys are
        treated as component names.

        Each component is resolved from the container (with automatic
        constructor injection), then ``initialize()`` is called on
        every component that supports it, in dependency order.

        Args:
            config: A Hydra ``DictConfig`` describing the pipeline.

        Returns:
            A fully resolved and initialised ``Pipeline``.

        Raises:
            PipelineValidationError: If validation fails.
            DependencyNotFoundError: If a dependency cannot be resolved.
            CircularDependencyError: If a cycle is detected.
        """
        component_names = self._extract_names(config)
        self.validate(config)

        # Build topological order
        dep_graph = self._build_dependency_graph(component_names)
        self._build_order = self._topological_sort(dep_graph)

        # Resolve all components
        components: Dict[str, Any] = {}
        for name in self._build_order:
            if self._container.contains(name):
                components[name] = self._container.resolve(name)

        # Lifecycle: initialize in dependency order
        for name in self._build_order:
            comp = components.get(name)
            if comp is not None and hasattr(comp, "initialize"):
                initializer = getattr(comp, "initialize")
                if callable(initializer):
                    initializer()
                    logger.info("Initialized: %s", name)

        self._pipeline = Pipeline(
            components=components,
            build_order=list(self._build_order),
        )
        logger.info("Pipeline built successfully")
        return self._pipeline

    def validate(self, config: DictConfig) -> None:
        """
        Validate the pipeline configuration.

        Checks:
          - All requested components are registered.
          - All transitive dependencies are available.
          - No circular dependencies exist.

        Args:
            config: The pipeline configuration.

        Raises:
            PipelineValidationError: If any check fails.
        """
        component_names = self._extract_names(config)

        # Check all components are registered
        missing = [
            n for n in component_names
            if not self._container.contains(n)
        ]
        if missing:
            raise PipelineValidationError(
                f"Components not registered: {missing}"
            )

        # Check transitive dependencies
        all_deps = self._collect_transitive_deps(component_names)
        missing_deps = [
            d for d in all_deps
            if not self._container.contains(d)
        ]
        if missing_deps:
            raise PipelineValidationError(
                f"Missing transitive dependencies: {missing_deps}"
            )

        # Check for cycles
        full_names = list(set(component_names) | all_deps)
        dep_graph = self._build_dependency_graph(full_names)
        try:
            self._topological_sort(dep_graph)
        except CircularDependencyError:
            raise PipelineValidationError(
                "Circular dependency detected in the pipeline."
            )

        logger.info("Pipeline validation passed")

    def visualize(self) -> str:
        """
        Produce a textual representation of the dependency graph.

        Returns:
            A multi-line string showing each component and its deps.
        """
        if not self._build_order:
            return "Pipeline: (empty)"

        lines = ["Pipeline Dependency Graph:", ""]
        for name in self._build_order:
            deps = self._container.get_dependencies(name)
            if deps:
                dep_str = ", ".join(deps)
                lines.append(f"  {name} ← [{dep_str}]")
            else:
                lines.append(f"  {name} ← (no dependencies)")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """
        Shut down all components in reverse dependency order.

        Calls ``shutdown()`` on every component that supports it.
        Errors during shutdown are logged but do not prevent
        subsequent components from shutting down.
        """
        if self._pipeline is None:
            logger.warning("shutdown() called on unbuilt pipeline")
            return

        for name in reversed(self._build_order):
            comp = self._pipeline.components.get(name)
            if comp is not None and hasattr(comp, "shutdown"):
                shutdown_fn = getattr(comp, "shutdown")
                if callable(shutdown_fn):
                    try:
                        shutdown_fn()
                        logger.info("Shutdown: %s", name)
                    except Exception:
                        logger.exception(
                            "Error shutting down %s", name
                        )

        self._pipeline = None
        self._build_order = []
        logger.info("Pipeline shutdown complete")

    # ── Private helpers ─────────────────────────────────────────

    @staticmethod
    def _extract_names(config: DictConfig) -> List[str]:
        """Extract component names from config."""
        if "components" in config:
            raw = config.components
            if isinstance(raw, (list, tuple)):
                return [str(n) for n in raw]
        return [str(k) for k in config]

    def _build_dependency_graph(
        self, names: Sequence[str]
    ) -> Dict[str, Set[str]]:
        """Build adjacency dict: node → set of predecessors."""
        graph: Dict[str, Set[str]] = {}
        for name in names:
            if self._container.contains(name):
                deps = self._container.get_dependencies(name)
                graph[name] = set(deps)
            else:
                graph[name] = set()
        return graph

    def _collect_transitive_deps(
        self, names: Sequence[str]
    ) -> Set[str]:
        """Collect all transitive dependencies recursively."""
        visited: Set[str] = set()
        stack = list(names)
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if self._container.contains(current):
                for dep in self._container.get_dependencies(current):
                    if dep not in visited:
                        stack.append(dep)
        return visited - set(names)

    @staticmethod
    def _topological_sort(
        graph: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Topologically sort the dependency graph.

        Args:
            graph: Adjacency dict (node → predecessors).

        Returns:
            Ordered list of component names.

        Raises:
            CircularDependencyError: If a cycle is detected.
        """
        sorter: TopologicalSorter[str] = TopologicalSorter(graph)
        try:
            return list(sorter.static_order())
        except CycleError as exc:
            raise CircularDependencyError(
                f"Circular dependency in graph: {exc}"
            ) from exc
