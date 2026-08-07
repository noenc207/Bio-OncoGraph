"""Tests for the graph statistics module."""

import networkx as nx
import pytest

from bio_oncograph.ai.graph import statistics


def test_stats_empty_graph() -> None:
    """Test statistics on an empty graph."""
    graph = nx.Graph()
    stats = statistics.compute_statistics(graph)
    assert stats["nodes"] == 0
    assert stats["edges"] == 0
    assert stats["density"] == 0.0
    assert stats["avg_degree"] == 0.0


def test_stats_undirected_graph() -> None:
    """Test statistics on an undirected graph."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    stats = statistics.compute_statistics(graph)
    assert stats["nodes"] == 3
    assert stats["edges"] == 3
    assert stats["density"] == 1.0
    assert stats["avg_degree"] == 2.0


def test_stats_directed_graph() -> None:
    """Test statistics on a directed graph."""
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 3), (3, 1)])
    stats = statistics.compute_statistics(graph)
    assert stats["nodes"] == 3
    assert stats["edges"] == 3
    assert stats["density"] == 0.5
    assert stats["avg_degree"] == 1.0


def test_stats_connected_components() -> None:
    """Test connected components statistic."""
    graph = nx.Graph()
    graph.add_edges_from([(1, 2), (3, 4)])
    stats = statistics.compute_statistics(graph)
    assert stats["connected_components"] == 2


def test_stats_isolated_nodes() -> None:
    """Test isolated nodes statistic."""
    graph = nx.Graph()
    graph.add_node(1)
    graph.add_node(2)
    stats = statistics.compute_statistics(graph)
    assert stats["isolated_nodes"] == 2


def test_stats_invalid_graph_type() -> None:
    """Test statistics with invalid graph type raises ValueError."""
    with pytest.raises(ValueError):
        statistics.compute_statistics("not_a_graph")


def test_stats_large_graph() -> None:
    """Test statistics on a larger graph."""
    graph = nx.complete_graph(10)
    stats = statistics.compute_statistics(graph)
    assert stats["nodes"] == 10
    assert stats["edges"] == 45
