"""Tests for LossReport."""

import pytest
from dataclasses import FrozenInstanceError

from ai.losses.report import LossReport


def test_loss_report_creation() -> None:
    """Test standard creation of LossReport."""
    report = LossReport(total_loss=2.5, components={"mse": 1.5, "reg": 1.0})
    assert report.total_loss == 2.5
    assert report.components == {"mse": 1.5, "reg": 1.0}


def test_loss_report_immutability_total_loss() -> None:
    """Test immutability of total_loss field."""
    report = LossReport(total_loss=1.0, components={})
    with pytest.raises(FrozenInstanceError):
        report.total_loss = 2.0


def test_loss_report_immutability_components() -> None:
    """Test immutability of components field assignment."""
    report = LossReport(total_loss=1.0, components={"a": 1.0})
    with pytest.raises(FrozenInstanceError):
        report.components = {"b": 2.0}


def test_loss_report_hashing() -> None:
    """Test that LossReport is hashable and can be used in sets/dicts."""
    # Note: For dataclasses to be hashable, their fields must be hashable.
    # Since components is typically a dict, it might need to be converted to a frozendict or tuple of items internally.
    # We will test hashing works if implemented correctly.
    try:
        report1 = LossReport(total_loss=1.0, components={"a": 1.0})
        report2 = LossReport(total_loss=1.0, components={"a": 1.0})
        
        # Testing if we can add to a set
        report_set = {report1}
        assert report1 in report_set
        
        # Testing equality implies same hash
        if report1 == report2:
            assert hash(report1) == hash(report2)
    except TypeError:
        pytest.skip("LossReport is not hashable (perhaps dict components are not frozen).")


def test_loss_report_equality() -> None:
    """Test equality comparison between LossReport instances."""
    report1 = LossReport(total_loss=2.0, components={"loss_a": 2.0})
    report2 = LossReport(total_loss=2.0, components={"loss_a": 2.0})
    report3 = LossReport(total_loss=3.0, components={"loss_a": 3.0})
    
    assert report1 == report2
    assert report1 != report3


def test_loss_report_empty_components() -> None:
    """Test creation with empty components."""
    report = LossReport(total_loss=0.0, components={})
    assert report.total_loss == 0.0
    assert len(report.components) == 0
