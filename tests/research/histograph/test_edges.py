import pytest
import numpy as np

from bio_oncograph.research.histograph.edges import (
    KNNEdgeStrategy,
    RadiusEdgeStrategy,
    FullyConnectedEdgeStrategy
)


def test_knn_edge_strategy_init() -> None:
    """Test KNNEdgeStrategy initialization."""
    strategy = KNNEdgeStrategy(k=5)
    assert strategy.k == 5


def test_knn_edge_strategy_invalid_k() -> None:
    """Test KNNEdgeStrategy raises on invalid k."""
    with pytest.raises(ValueError):
        KNNEdgeStrategy(k=0)


def test_knn_edge_strategy_compute_empty() -> None:
    """Test KNNEdgeStrategy with empty points."""
    strategy = KNNEdgeStrategy(k=3)
    pts = np.empty((0, 2))
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_knn_edge_strategy_compute_single_node() -> None:
    """Test KNNEdgeStrategy with one point."""
    strategy = KNNEdgeStrategy(k=2)
    pts = np.array([[0.0, 0.0]])
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_knn_edge_strategy_compute_valid() -> None:
    """Test KNNEdgeStrategy computes correct edges."""
    strategy = KNNEdgeStrategy(k=1)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    edges = strategy.compute_edges(pts)
    assert edges.shape[0] == 2
    assert edges.shape[1] == 3


def test_radius_edge_strategy_init() -> None:
    """Test RadiusEdgeStrategy initialization."""
    strategy = RadiusEdgeStrategy(radius=10.0)
    assert strategy.radius == 10.0


def test_radius_edge_strategy_invalid_radius() -> None:
    """Test RadiusEdgeStrategy raises on invalid radius."""
    with pytest.raises(ValueError):
        RadiusEdgeStrategy(radius=-1.0)


def test_radius_edge_strategy_compute_empty() -> None:
    """Test RadiusEdgeStrategy with empty points."""
    strategy = RadiusEdgeStrategy(radius=5.0)
    pts = np.empty((0, 2))
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_radius_edge_strategy_compute_single_node() -> None:
    """Test RadiusEdgeStrategy with one point."""
    strategy = RadiusEdgeStrategy(radius=5.0)
    pts = np.array([[1.0, 1.0]])
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_radius_edge_strategy_compute_within_radius() -> None:
    """Test RadiusEdgeStrategy connects points within radius."""
    strategy = RadiusEdgeStrategy(radius=2.0)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    edges = strategy.compute_edges(pts)
    # 0 and 1 should connect to each other. 2 is isolated.
    assert edges.shape[1] == 2


def test_radius_edge_strategy_symmetry() -> None:
    """Test RadiusEdgeStrategy output is symmetric."""
    strategy = RadiusEdgeStrategy(radius=5.0)
    pts = np.random.rand(10, 2) * 10
    edges = strategy.compute_edges(pts)
    edge_set = set(zip(edges[0], edges[1]))
    for u, v in edge_set:
        assert (v, u) in edge_set


def test_fully_connected_edge_strategy_init() -> None:
    """Test FullyConnectedEdgeStrategy initialization."""
    strategy = FullyConnectedEdgeStrategy()
    assert strategy is not None


def test_fully_connected_edge_strategy_empty() -> None:
    """Test FullyConnectedEdgeStrategy on empty points."""
    strategy = FullyConnectedEdgeStrategy()
    pts = np.empty((0, 2))
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_fully_connected_edge_strategy_single() -> None:
    """Test FullyConnectedEdgeStrategy on single point."""
    strategy = FullyConnectedEdgeStrategy()
    pts = np.array([[0.0, 0.0]])
    edges = strategy.compute_edges(pts)
    assert edges.shape == (2, 0)


def test_fully_connected_edge_strategy_compute() -> None:
    """Test FullyConnectedEdgeStrategy connects all pairs."""
    strategy = FullyConnectedEdgeStrategy()
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    edges = strategy.compute_edges(pts)
    # N=3 -> 3*2 = 6 directed edges (no self loops)
    assert edges.shape[1] == 6


def test_fully_connected_no_self_loops() -> None:
    """Test FullyConnectedEdgeStrategy does not produce self loops."""
    strategy = FullyConnectedEdgeStrategy()
    pts = np.random.rand(5, 2)
    edges = strategy.compute_edges(pts)
    assert not np.any(edges[0] == edges[1])


def test_knn_no_self_loops() -> None:
    """Test KNNEdgeStrategy does not produce self loops."""
    strategy = KNNEdgeStrategy(k=2)
    pts = np.random.rand(10, 2)
    edges = strategy.compute_edges(pts)
    assert not np.any(edges[0] == edges[1])


def test_radius_no_self_loops() -> None:
    """Test RadiusEdgeStrategy does not produce self loops."""
    strategy = RadiusEdgeStrategy(radius=10.0)
    pts = np.random.rand(10, 2)
    edges = strategy.compute_edges(pts)
    assert not np.any(edges[0] == edges[1])


def test_strategies_large_graph() -> None:
    """Test strategies scale to larger number of points without crash."""
    pts = np.random.rand(1000, 2) * 100
    strat = KNNEdgeStrategy(k=5)
    edges = strat.compute_edges(pts)
    assert edges.shape[1] == 5000
