"""Tests for the GraphReport class."""

import pytest

from bio_oncograph.ai.graph.report import GraphReport


def test_report_creation() -> None:
    """Test the creation of a GraphReport instance."""
    report = GraphReport(nodes=10, edges=15, density=0.5)
    assert report.nodes == 10
    assert report.edges == 15
    assert report.density == 0.5


def test_report_immutability() -> None:
    """Test that GraphReport instances are immutable."""
    report = GraphReport(nodes=10, edges=15, density=0.5)
    with pytest.raises(Exception):
        # Depending on implementation, could be AttributeError or dataclass FrozenInstanceError
        report.nodes = 20  # type: ignore


def test_report_hash() -> None:
    """Test that GraphReport instances are hashable."""
    report1 = GraphReport(nodes=10, edges=15, density=0.5)
    report2 = GraphReport(nodes=10, edges=15, density=0.5)
    report3 = GraphReport(nodes=11, edges=15, density=0.5)
    
    assert hash(report1) == hash(report2)
    assert hash(report1) != hash(report3)
    
    # Should be usable in a set
    report_set = {report1, report2, report3}
    assert len(report_set) == 2


def test_report_equality() -> None:
    """Test equality comparison between GraphReport instances."""
    report1 = GraphReport(nodes=10, edges=15, density=0.5)
    report2 = GraphReport(nodes=10, edges=15, density=0.5)
    report3 = GraphReport(nodes=20, edges=15, density=0.5)
    
    assert report1 == report2
    assert report1 != report3
    assert report1 != "not_a_report"
