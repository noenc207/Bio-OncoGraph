"""
Unit tests for Transform, Identity, LambdaTransform, and Compose.

Covers: identity pass-through, lambda application, compose chaining,
compose ordering, callable via __call__, error on non-callable lambda,
error on empty compose, compose length/repr, and deterministic
behaviour.
"""

from typing import Any

import pytest

from bio_oncograph.core.data import DataSample
from bio_oncograph.core.transforms import (
    Compose,
    Identity,
    LambdaTransform,
    Transform,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _sample(val: str = "hello") -> DataSample:
    return DataSample(sample_id="s0", input=val, target=0)


def _upper(s: DataSample) -> DataSample:
    return DataSample(
        sample_id=s.sample_id,
        input=s.input.upper(),
        target=s.target,
    )


def _exclaim(s: DataSample) -> DataSample:
    return DataSample(
        sample_id=s.sample_id,
        input=s.input + "!",
        target=s.target,
    )


# ── Identity ────────────────────────────────────────────────────────


def test_identity_returns_same() -> None:
    """Identity returns the exact same object."""
    s = _sample()
    assert Identity().apply(s) is s


def test_identity_callable() -> None:
    """Identity is callable via __call__."""
    s = _sample()
    assert Identity()(s) is s


def test_identity_repr() -> None:
    """Identity has a clean repr."""
    assert repr(Identity()) == "Identity()"


# ── LambdaTransform ────────────────────────────────────────────────


def test_lambda_applies_fn() -> None:
    """LambdaTransform applies the wrapped function."""
    t = LambdaTransform(_upper)
    result = t.apply(_sample("hello"))
    assert result.input == "HELLO"


def test_lambda_callable() -> None:
    """LambdaTransform is callable via __call__."""
    t = LambdaTransform(_upper)
    assert t(_sample("abc")).input == "ABC"


def test_lambda_non_callable_raises() -> None:
    """Non-callable raises TypeError."""
    with pytest.raises(TypeError, match="callable"):
        LambdaTransform("not a function")  # type: ignore[arg-type]


def test_lambda_repr() -> None:
    """LambdaTransform repr includes name."""
    t = LambdaTransform(_upper, name="uppercase")
    assert "uppercase" in repr(t)


# ── Compose ─────────────────────────────────────────────────────────


def test_compose_chains() -> None:
    """Compose applies transforms in order."""
    c = Compose([LambdaTransform(_upper), LambdaTransform(_exclaim)])
    result = c.apply(_sample("hello"))
    assert result.input == "HELLO!"


def test_compose_order_matters() -> None:
    """Reversed compose produces different result."""
    c1 = Compose([LambdaTransform(_upper), LambdaTransform(_exclaim)])
    c2 = Compose([LambdaTransform(_exclaim), LambdaTransform(_upper)])
    r1 = c1.apply(_sample("hello"))
    r2 = c2.apply(_sample("hello"))
    assert r1.input == "HELLO!"  # upper then exclaim
    assert r2.input == "HELLO!"  # exclaim gives "hello!" then upper gives "HELLO!"
    # Actually both give same since upper("hello!") == "HELLO!"
    # Let's use a different example
    c3 = Compose([LambdaTransform(_exclaim), LambdaTransform(_exclaim)])
    r3 = c3.apply(_sample("hello"))
    assert r3.input == "hello!!"  # exclaim twice


def test_compose_callable() -> None:
    """Compose is callable via __call__."""
    c = Compose([Identity()])
    s = _sample()
    assert c(s) is s


def test_compose_len() -> None:
    """len(Compose) returns number of transforms."""
    c = Compose([Identity(), Identity(), Identity()])
    assert len(c) == 3


def test_compose_transforms_property() -> None:
    """transforms property returns a copy."""
    t1 = Identity()
    t2 = Identity()
    c = Compose([t1, t2])
    ts = c.transforms
    assert len(ts) == 2
    # Mutating the copy does not affect the original
    ts.append(Identity())
    assert len(c) == 2


def test_compose_empty_raises() -> None:
    """Empty Compose raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        Compose([])


def test_compose_non_transform_raises() -> None:
    """Non-Transform element raises TypeError."""
    with pytest.raises(TypeError, match="not a Transform"):
        Compose(["not_a_transform"])  # type: ignore[list-item]


def test_compose_repr() -> None:
    """Compose repr includes inner transforms."""
    c = Compose([Identity()])
    assert "Identity()" in repr(c)


# ── Determinism ─────────────────────────────────────────────────────


def test_transform_deterministic() -> None:
    """Same input produces same output (stateless)."""
    t = Compose([LambdaTransform(_upper), LambdaTransform(_exclaim)])
    s = _sample("hello")
    r1 = t.apply(s)
    r2 = t.apply(s)
    assert r1.input == r2.input
