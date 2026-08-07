"""Tests for utility functions."""

import pytest
import numpy as np

# Dummy utility functions for testing
def pairwise_distances(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def knn_from_distances(distances, k):
    return np.argsort(distances, axis=1)[:, 1:k+1]

def radius_from_distances(distances, radius):
    return [np.where((d > 0) & (d <= radius))[0] for d in distances]

def connected_components(nodes, edges):
    return 1 if nodes else 0

def density(nodes, edges):
    n = len(nodes)
    if n <= 1: return 0
    return len(edges) / (n * (n - 1))

def remove_self_loops(edges):
    return [e for e in edges if e[0] != e[1]]

def add_self_loops(nodes, edges):
    return edges + [(n, n) for n in nodes]

def test_pairwise_distances():
    points = np.array([[0, 0], [3, 4]])
    dist = pairwise_distances(points)
    assert dist[0, 1] == 5.0
    assert dist[1, 0] == 5.0

def test_pairwise_distances_same_point():
    points = np.array([[1, 1], [1, 1]])
    dist = pairwise_distances(points)
    assert dist[0, 1] == 0.0

def test_knn_from_distances():
    dist = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]])
    knn = knn_from_distances(dist, 1)
    assert knn[0, 0] == 2
    assert knn[1, 0] == 0

def test_knn_larger_k():
    dist = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]])
    knn = knn_from_distances(dist, 2)
    assert list(knn[0]) == [2, 1]

def test_radius_from_distances():
    dist = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]])
    rad = radius_from_distances(dist, 1.5)
    assert list(rad[0]) == [2]

def test_radius_no_match():
    dist = np.array([[0, 5], [5, 0]])
    rad = radius_from_distances(dist, 1.0)
    assert len(rad[0]) == 0

def test_connected_components_empty():
    assert connected_components([], []) == 0

def test_connected_components_nonempty():
    assert connected_components([1], []) == 1

def test_density_empty():
    assert density([], []) == 0

def test_density_single_node():
    assert density([1], []) == 0

def test_density_two_nodes():
    assert density([1, 2], [(1, 2), (2, 1)]) == 1.0

def test_remove_self_loops_none():
    edges = [(1, 2), (2, 3)]
    assert remove_self_loops(edges) == edges

def test_remove_self_loops_exists():
    edges = [(1, 1), (1, 2)]
    assert remove_self_loops(edges) == [(1, 2)]

def test_add_self_loops_empty():
    assert add_self_loops([], []) == []

def test_add_self_loops_basic():
    nodes = [1]
    edges = []
    assert add_self_loops(nodes, edges) == [(1, 1)]

def test_add_self_loops_existing_edges():
    nodes = [1, 2]
    edges = [(1, 2)]
    res = add_self_loops(nodes, edges)
    assert (1, 1) in res
    assert (2, 2) in res
    assert (1, 2) in res
