"""
Unit tests for ComponentFactory.

Coverage targets:
  - create with valid kwargs
  - create with extra kwargs (silently filtered)
  - create with missing mandatory args
  - create_from_config success
  - create_from_config missing 'name' key
  - create_many
  - constructor with **kwargs passthrough
  - invalid component name
  - factory repr / registry access
"""

from typing import Any

import pytest
from omegaconf import OmegaConf

from bio_oncograph.core.factory import ComponentFactory
from bio_oncograph.core.registry import ComponentRegistry


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def registry() -> ComponentRegistry:
    """Fresh registry with pre-registered dummy classes."""
    reg = ComponentRegistry(name="test_factory")
    reg.register("simple", SimpleComponent)
    reg.register("defaulted", DefaultedComponent)
    reg.register("var_kw", VarKwargsComponent)
    return reg


@pytest.fixture()
def factory(registry: ComponentRegistry) -> ComponentFactory:
    """Factory backed by the test registry."""
    return ComponentFactory(registry)


# ── Dummy components ────────────────────────────────────────────────


class SimpleComponent:
    """Component with one mandatory and one optional arg."""

    def __init__(self, dim: int, rate: float = 0.1) -> None:
        self.dim = dim
        self.rate = rate


class DefaultedComponent:
    """Component where all args have defaults."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta


class VarKwargsComponent:
    """Component whose constructor accepts **kwargs."""

    def __init__(self, base: int, **kwargs: Any) -> None:
        self.base = base
        self.extras = kwargs


# ── create ──────────────────────────────────────────────────────────


def test_create_success(factory: ComponentFactory) -> None:
    """create instantiates with correct kwargs."""
    obj = factory.create("simple", dim=128, rate=0.5)
    assert isinstance(obj, SimpleComponent)
    assert obj.dim == 128
    assert obj.rate == 0.5


def test_create_uses_defaults(factory: ComponentFactory) -> None:
    """Omitted optional args fall back to constructor defaults."""
    obj = factory.create("simple", dim=64)
    assert obj.rate == 0.1


def test_create_filters_extra_kwargs(factory: ComponentFactory) -> None:
    """Extra kwargs not in the constructor are silently dropped."""
    obj = factory.create(
        "simple", dim=32, rate=0.2, unknown_key="ignored"
    )
    assert obj.dim == 32
    assert not hasattr(obj, "unknown_key")


def test_create_missing_mandatory_raises(
    factory: ComponentFactory,
) -> None:
    """Missing mandatory arg raises TypeError with clear message."""
    with pytest.raises(TypeError, match="Missing required.*dim"):
        factory.create("simple", rate=0.5)


def test_create_unknown_name_raises(
    factory: ComponentFactory,
) -> None:
    """Unknown component name raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        factory.create("nonexistent", dim=1)


def test_create_all_defaults(factory: ComponentFactory) -> None:
    """Component with all-default args can be created with no kwargs."""
    obj = factory.create("defaulted")
    assert obj.alpha == 1.0
    assert obj.beta == 0.5


def test_create_var_kwargs_passthrough(
    factory: ComponentFactory,
) -> None:
    """Components with **kwargs receive all extra keys."""
    obj = factory.create("var_kw", base=10, extra1="a", extra2=42)
    assert obj.base == 10
    assert obj.extras == {"extra1": "a", "extra2": 42}


# ── create_from_config ──────────────────────────────────────────────


def test_create_from_config_success(
    factory: ComponentFactory,
) -> None:
    """create_from_config resolves OmegaConf and instantiates."""
    cfg = OmegaConf.create({"name": "simple", "dim": 256, "rate": 0.3})
    obj = factory.create_from_config(cfg)
    assert isinstance(obj, SimpleComponent)
    assert obj.dim == 256
    assert obj.rate == 0.3


def test_create_from_config_extra_fields(
    factory: ComponentFactory,
) -> None:
    """Extra config fields not in the constructor are dropped."""
    cfg = OmegaConf.create({
        "name": "simple",
        "dim": 64,
        "extra_field": True,
    })
    obj = factory.create_from_config(cfg)
    assert obj.dim == 64


def test_create_from_config_missing_name_raises(
    factory: ComponentFactory,
) -> None:
    """Config without 'name' raises ValueError."""
    cfg = OmegaConf.create({"dim": 128})
    with pytest.raises(ValueError, match="name"):
        factory.create_from_config(cfg)


# ── create_many ─────────────────────────────────────────────────────


def test_create_many(factory: ComponentFactory) -> None:
    """create_many returns a list of instantiated components."""
    configs = [
        OmegaConf.create({"name": "simple", "dim": 32}),
        OmegaConf.create({"name": "defaulted", "alpha": 2.0}),
    ]
    objects = factory.create_many(configs)
    assert len(objects) == 2
    assert isinstance(objects[0], SimpleComponent)
    assert isinstance(objects[1], DefaultedComponent)
    assert objects[0].dim == 32
    assert objects[1].alpha == 2.0


# ── Factory properties ──────────────────────────────────────────────


def test_factory_registry_property(
    factory: ComponentFactory, registry: ComponentRegistry
) -> None:
    """Factory exposes its backing registry."""
    assert factory.registry is registry


# ── Registry isolation ──────────────────────────────────────────────


def test_separate_registries_isolated() -> None:
    """Two ComponentRegistry instances do not share state."""
    reg_a = ComponentRegistry(name="a")
    reg_b = ComponentRegistry(name="b")
    reg_a.register("only_in_a", SimpleComponent)
    assert reg_a.contains("only_in_a")
    assert not reg_b.contains("only_in_a")
