"""
Unit tests for ComponentRegistry and backward-compatible wrappers.

Coverage targets:
  - register / duplicate / unregister / contains / get / list / clear
  - thread safety (concurrent registration)
  - deterministic insertion ordering
  - backward-compat: register_module, build_from_config, load_plugins
  - error paths: unknown component, invalid type, missing config key
"""

import os
import threading
from typing import Any
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from bio_oncograph.core.exceptions import ConfigException
from bio_oncograph.core.registry import (
    ComponentRegistry,
    _get_or_create_group,
    _group_registries,
    build_from_config,
    load_plugins,
    register_module,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> ComponentRegistry:
    """Fresh, isolated registry instance for each test."""
    return ComponentRegistry(name="test")


@pytest.fixture(autouse=True)
def _clear_global_registries() -> Any:
    """Ensure module-level registries are clean between tests."""
    _group_registries.clear()
    yield
    _group_registries.clear()


# ── Dummy classes ───────────────────────────────────────────────────


class DummyA:
    def __init__(self, dim: int, rate: float = 0.0) -> None:
        self.dim = dim
        self.rate = rate


class DummyB:
    def __init__(self, size: int) -> None:
        self.size = size


# ── ComponentRegistry: core operations ──────────────────────────────


def test_register_success(registry: ComponentRegistry) -> None:
    """Registering a class succeeds and is retrievable."""
    registry.register("a", DummyA)
    assert registry.contains("a")
    assert registry.get("a") is DummyA


def test_register_duplicate_raises(registry: ComponentRegistry) -> None:
    """Duplicate registration raises ValueError."""
    registry.register("a", DummyA)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a", DummyB)


def test_register_non_type_raises(registry: ComponentRegistry) -> None:
    """Registering an instance (not a type) raises TypeError."""
    with pytest.raises(TypeError, match="Expected a class"):
        registry.register("bad", DummyA(dim=1))  # type: ignore[arg-type]


def test_unregister(registry: ComponentRegistry) -> None:
    """Unregister removes the component."""
    registry.register("a", DummyA)
    registry.unregister("a")
    assert not registry.contains("a")


def test_unregister_unknown_raises(registry: ComponentRegistry) -> None:
    """Unregistering an unknown name raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        registry.unregister("nonexistent")


def test_contains_false(registry: ComponentRegistry) -> None:
    """contains returns False for unregistered names."""
    assert not registry.contains("missing")


def test_get_unknown_raises(registry: ComponentRegistry) -> None:
    """Getting an unregistered name raises KeyError with available list."""
    registry.register("a", DummyA)
    with pytest.raises(KeyError, match="Available.*a"):
        registry.get("unknown")


def test_list_components_empty(registry: ComponentRegistry) -> None:
    """Empty registry returns empty list."""
    assert registry.list_components() == []


def test_list_components_ordered(registry: ComponentRegistry) -> None:
    """Components are listed in insertion order."""
    registry.register("b", DummyB)
    registry.register("a", DummyA)
    assert registry.list_components() == ["b", "a"]


def test_clear(registry: ComponentRegistry) -> None:
    """clear removes all components."""
    registry.register("a", DummyA)
    registry.register("b", DummyB)
    registry.clear()
    assert len(registry) == 0
    assert registry.list_components() == []


def test_len(registry: ComponentRegistry) -> None:
    """__len__ reflects the number of registered components."""
    assert len(registry) == 0
    registry.register("a", DummyA)
    assert len(registry) == 1


def test_in_operator(registry: ComponentRegistry) -> None:
    """The 'in' operator delegates to contains."""
    registry.register("a", DummyA)
    assert "a" in registry
    assert "b" not in registry


def test_repr(registry: ComponentRegistry) -> None:
    """repr includes name and component list."""
    registry.register("a", DummyA)
    r = repr(registry)
    assert "test" in r
    assert "a" in r


# ── Thread safety ───────────────────────────────────────────────────


def test_concurrent_registration(registry: ComponentRegistry) -> None:
    """Multiple threads can register without data corruption."""
    errors: list[Exception] = []

    def register_class(idx: int) -> None:
        try:
            # Create a unique class per thread
            cls = type(f"Dynamic{idx}", (), {})
            registry.register(f"comp_{idx}", cls)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=register_class, args=(i,))
        for i in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(registry) == 50


# ── Backward-compat: register_module ────────────────────────────────


def test_register_module_decorator() -> None:
    """@register_module decorator registers into group registries."""

    @register_module(name="dummy_model", group="model")
    class MyModel:
        pass

    group_reg = _get_or_create_group("model")
    assert group_reg.contains("dummy_model")
    assert group_reg.get("dummy_model") is MyModel


def test_register_module_duplicate_raises() -> None:
    """Duplicate @register_module raises ConfigException."""
    register_module(name="dup", group="test_group")(DummyA)
    with pytest.raises(ConfigException):
        register_module(name="dup", group="test_group")(DummyB)


# ── Backward-compat: build_from_config ──────────────────────────────


def test_build_from_config_success() -> None:
    """build_from_config instantiates from registry + config."""
    register_module(name="dummy", group="model")(DummyA)
    config = OmegaConf.create({
        "_target_name_": "dummy",
        "dim": 256,
        "rate": 0.5,
    })
    instance = build_from_config(config, group="model")
    assert isinstance(instance, DummyA)
    assert instance.dim == 256
    assert instance.rate == 0.5


def test_build_from_config_missing_target_key() -> None:
    """Missing _target_name_ raises ConfigException."""
    config = OmegaConf.create({"dim": 256})
    with pytest.raises(ConfigException, match="_target_name_"):
        build_from_config(config, group="model")


def test_build_from_config_unknown_target() -> None:
    """Unknown target raises ConfigException."""
    config = OmegaConf.create({"_target_name_": "ghost"})
    with pytest.raises(ConfigException, match="not found"):
        build_from_config(config, group="model")


# ── Backward-compat: load_plugins ───────────────────────────────────


@patch("bio_oncograph.core.registry.importlib.import_module")
@patch.dict(os.environ, {"BIOONCO_PLUGINS": "pkg.a, pkg.b"})
def test_load_plugins_imports(mock_import: Any) -> None:
    """load_plugins imports each comma-separated module."""
    load_plugins()
    assert mock_import.call_count == 2
    mock_import.assert_any_call("pkg.a")
    mock_import.assert_any_call("pkg.b")


@patch.dict(os.environ, {"BIOONCO_PLUGINS": ""})
def test_load_plugins_empty_noop() -> None:
    """Empty BIOONCO_PLUGINS is a no-op."""
    load_plugins()  # should not raise
