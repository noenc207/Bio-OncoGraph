"""
Unit tests for PipelineBuilder and Pipeline.

Covers: build from config, lifecycle initialize/shutdown ordering,
validate missing deps, validate circular deps, visualize, shutdown
error handling, and deterministic build order.
"""

from typing import Any, List
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from bio_oncograph.core.container import DIContainer
from bio_oncograph.core.exceptions import (
    PipelineValidationError,
)
from bio_oncograph.core.pipeline import Pipeline, PipelineBuilder


# ── Lifecycle tracking helpers ──────────────────────────────────────

_lifecycle_log: List[str] = []


class Engine:
    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        _lifecycle_log.append("Engine.initialize")

    def shutdown(self) -> None:
        _lifecycle_log.append("Engine.shutdown")


class Storage:
    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        _lifecycle_log.append("Storage.initialize")

    def shutdown(self) -> None:
        _lifecycle_log.append("Storage.shutdown")


class Processor:
    """Depends on engine and storage."""

    def __init__(self, engine: Any, storage: Any) -> None:
        self.engine = engine
        self.storage = storage

    def initialize(self) -> None:
        _lifecycle_log.append("Processor.initialize")

    def shutdown(self) -> None:
        _lifecycle_log.append("Processor.shutdown")


class SimpleComp:
    """No lifecycle hooks."""

    def __init__(self) -> None:
        self.value = 42


class CycleX:
    def __init__(self, cycle_y: Any) -> None:
        pass


class CycleY:
    def __init__(self, cycle_x: Any) -> None:
        pass


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_lifecycle_log() -> Any:
    """Clear lifecycle log between tests."""
    _lifecycle_log.clear()
    yield
    _lifecycle_log.clear()


@pytest.fixture()
def container() -> DIContainer:
    """Container with pre-registered test components."""
    c = DIContainer()
    c.register("engine", Engine)
    c.register("storage", Storage)
    c.register("processor", Processor)
    return c


@pytest.fixture()
def builder(container: DIContainer) -> PipelineBuilder:
    """PipelineBuilder backed by the test container."""
    return PipelineBuilder(container)


# ── Pipeline dataclass ──────────────────────────────────────────────


def test_pipeline_get() -> None:
    """Pipeline.get retrieves a component."""
    p = Pipeline(components={"a": 1, "b": 2}, build_order=["a", "b"])
    assert p.get("a") == 1


def test_pipeline_get_missing_raises() -> None:
    """Pipeline.get raises KeyError for unknown name."""
    p = Pipeline(components={}, build_order=[])
    with pytest.raises(KeyError, match="not in pipeline"):
        p.get("ghost")


def test_pipeline_contains() -> None:
    """'in' operator works on Pipeline."""
    p = Pipeline(components={"a": 1}, build_order=["a"])
    assert "a" in p
    assert "b" not in p


def test_pipeline_len() -> None:
    """len(Pipeline) returns component count."""
    p = Pipeline(components={"a": 1, "b": 2}, build_order=["a", "b"])
    assert len(p) == 2


# ── PipelineBuilder.build ──────────────────────────────────────────


def test_build_resolves_components(
    builder: PipelineBuilder,
) -> None:
    """build resolves all named components."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    pipeline = builder.build(config)
    assert "engine" in pipeline
    assert "storage" in pipeline
    assert "processor" in pipeline
    assert isinstance(pipeline.get("processor"), Processor)


def test_build_injects_dependencies(
    builder: PipelineBuilder,
) -> None:
    """Processor's constructor deps are injected."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    pipeline = builder.build(config)
    proc = pipeline.get("processor")
    assert isinstance(proc.engine, Engine)
    assert isinstance(proc.storage, Storage)


def test_build_calls_initialize_in_order(
    builder: PipelineBuilder,
) -> None:
    """initialize() is called in dependency order."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    builder.build(config)
    # Engine and Storage have no deps → initialised first
    # Processor depends on both → initialised last
    assert "Processor.initialize" in _lifecycle_log
    proc_idx = _lifecycle_log.index("Processor.initialize")
    eng_idx = _lifecycle_log.index("Engine.initialize")
    stor_idx = _lifecycle_log.index("Storage.initialize")
    assert proc_idx > eng_idx
    assert proc_idx > stor_idx


def test_build_with_dict_keys(container: DIContainer) -> None:
    """build works when config keys are component names."""
    container.register("simple", SimpleComp)
    builder = PipelineBuilder(container)
    config = OmegaConf.create({"simple": {}})
    pipeline = builder.build(config)
    assert isinstance(pipeline.get("simple"), SimpleComp)


# ── PipelineBuilder.validate ───────────────────────────────────────


def test_validate_success(builder: PipelineBuilder) -> None:
    """Validation passes for valid config."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    builder.validate(config)  # Should not raise


def test_validate_missing_component(
    builder: PipelineBuilder,
) -> None:
    """Validation fails if a component is not registered."""
    config = OmegaConf.create({"components": ["ghost"]})
    with pytest.raises(PipelineValidationError, match="not registered"):
        builder.validate(config)


def test_validate_circular_dependency() -> None:
    """Validation detects circular dependencies."""
    c = DIContainer()
    c.register("cycle_x", CycleX)
    c.register("cycle_y", CycleY)
    builder = PipelineBuilder(c)
    config = OmegaConf.create(
        {"components": ["cycle_x", "cycle_y"]}
    )
    with pytest.raises(PipelineValidationError, match="Circular"):
        builder.validate(config)


# ── PipelineBuilder.shutdown ───────────────────────────────────────


def test_shutdown_reverse_order(builder: PipelineBuilder) -> None:
    """shutdown calls shutdown() in reverse dependency order."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    builder.build(config)
    _lifecycle_log.clear()
    builder.shutdown()

    # Processor depends on Engine+Storage → shuts down first
    assert _lifecycle_log[0] == "Processor.shutdown"


def test_shutdown_unbuilt_pipeline_is_noop(
    builder: PipelineBuilder,
) -> None:
    """shutdown on unbuilt pipeline logs warning but doesn't crash."""
    builder.shutdown()  # Should not raise


def test_shutdown_error_does_not_block_others(
    container: DIContainer,
) -> None:
    """Errors during one shutdown don't prevent others."""

    class Broken:
        def initialize(self) -> None:
            pass

        def shutdown(self) -> None:
            raise RuntimeError("boom")

    container.register("broken", Broken)
    builder = PipelineBuilder(container)
    config = OmegaConf.create(
        {"components": ["engine", "broken"]}
    )
    builder.build(config)
    _lifecycle_log.clear()
    builder.shutdown()
    # Engine.shutdown should still be called
    assert "Engine.shutdown" in _lifecycle_log


# ── PipelineBuilder.visualize ──────────────────────────────────────


def test_visualize_empty(builder: PipelineBuilder) -> None:
    """visualize on empty pipeline returns placeholder."""
    assert "empty" in builder.visualize()


def test_visualize_after_build(builder: PipelineBuilder) -> None:
    """visualize shows dependency relationships."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    builder.build(config)
    viz = builder.visualize()
    assert "processor" in viz
    assert "engine" in viz


# ── Deterministic ordering ─────────────────────────────────────────


def test_deterministic_build_order(
    builder: PipelineBuilder,
) -> None:
    """Two identical builds produce the same order."""
    config = OmegaConf.create(
        {"components": ["engine", "storage", "processor"]}
    )
    p1 = builder.build(config)
    order1 = list(p1.build_order)

    builder2 = PipelineBuilder(builder.container)
    p2 = builder2.build(config)
    order2 = list(p2.build_order)
    assert order1 == order2


# ── Builder property ───────────────────────────────────────────────


def test_builder_container_property(
    builder: PipelineBuilder, container: DIContainer
) -> None:
    """Builder exposes its container."""
    assert builder.container is container
