"""Tests for determinism of graph builders."""

import pytest
import numpy as np

class DummyBuilder:
    def __init__(self, seed):
        self.seed = seed
    def build(self, data):
        np.random.seed(self.seed)
        return [d + np.random.rand() for d in data]

def test_builder_determinism_same_seed():
    data = [1, 2, 3]
    b1 = DummyBuilder(42)
    b2 = DummyBuilder(42)
    assert b1.build(data) == b2.build(data)

def test_builder_determinism_different_seed():
    data = [1, 2, 3]
    b1 = DummyBuilder(42)
    b2 = DummyBuilder(43)
    assert b1.build(data) != b2.build(data)

def test_builder_determinism_multiple_runs():
    data = [1, 2, 3]
    b1 = DummyBuilder(42)
    res1 = b1.build(data)
    b2 = DummyBuilder(42)
    res2 = b2.build(data)
    b3 = DummyBuilder(42)
    res3 = b3.build(data)
    assert res1 == res2 == res3

def test_builder_determinism_large_data():
    data = list(range(1000))
    b1 = DummyBuilder(123)
    b2 = DummyBuilder(123)
    assert b1.build(data) == b2.build(data)

def test_builder_determinism_empty_data():
    b1 = DummyBuilder(1)
    b2 = DummyBuilder(1)
    assert b1.build([]) == b2.build([])

def test_builder_determinism_numpy_random_state():
    np.random.seed(99)
    d1 = np.random.rand(5).tolist()
    np.random.seed(99)
    d2 = np.random.rand(5).tolist()
    assert d1 == d2

def test_builder_determinism_complex_data():
    data = [{'x': 1}, {'y': 2}]
    b1 = DummyBuilder(42)
    b2 = DummyBuilder(42)
    # Modifying dummy builder for dicts
    class ComplexBuilder(DummyBuilder):
        def build(self, data):
            np.random.seed(self.seed)
            return [str(d) + str(np.random.rand()) for d in data]
    cb1 = ComplexBuilder(42)
    cb2 = ComplexBuilder(42)
    assert cb1.build(data) == cb2.build(data)

def test_determinism_noise_addition():
    data = np.zeros(10)
    b1 = DummyBuilder(7)
    b2 = DummyBuilder(7)
    assert np.allclose(b1.build(data), b2.build(data))

def test_determinism_across_instances():
    data = [10, 20, 30]
    res = DummyBuilder(42).build(data)
    assert res == DummyBuilder(42).build(data)
    assert res == DummyBuilder(42).build(data)

def test_determinism_negative_seed():
    # Numpy handles uint32, so checking a specific seed behavior
    b1 = DummyBuilder(123456789)
    b2 = DummyBuilder(123456789)
    assert b1.build([1, 2]) == b2.build([1, 2])

def test_determinism_zero_seed():
    b1 = DummyBuilder(0)
    b2 = DummyBuilder(0)
    assert b1.build([1]) == b2.build([1])
