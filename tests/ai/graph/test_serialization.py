"""Tests for graph serialization."""

import os
from typing import Generator

import networkx as nx
import pytest

from bio_oncograph.ai.graph import serialization


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Fixture providing a sample graph."""
    graph = nx.Graph()
    graph.add_node(1, feature="A")
    graph.add_node(2, feature="B")
    graph.add_edge(1, 2, weight=1.5)
    return graph


def test_serialize_graph(sample_graph: nx.Graph, tmp_path: pytest.TempPathFactory) -> None:
    """Test serializing a graph to a file."""
    file_path = os.path.join(tmp_path, "test.pkl")
    serialization.serialize_graph(sample_graph, file_path)
    assert os.path.exists(file_path)


def test_deserialize_graph(sample_graph: nx.Graph, tmp_path: pytest.TempPathFactory) -> None:
    """Test deserializing a graph from a file."""
    file_path = os.path.join(tmp_path, "test.pkl")
    serialization.serialize_graph(sample_graph, file_path)
    graph = serialization.deserialize_graph(file_path)
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes) == 2


def test_roundtrip_serialization(sample_graph: nx.Graph, tmp_path: pytest.TempPathFactory) -> None:
    """Test that a graph can be serialized and deserialized without data loss."""
    file_path = os.path.join(tmp_path, "test.pkl")
    serialization.serialize_graph(sample_graph, file_path)
    loaded_graph = serialization.deserialize_graph(file_path)
    assert set(sample_graph.nodes) == set(loaded_graph.nodes)
    assert set(sample_graph.edges) == set(loaded_graph.edges)


def test_serialization_preserves_features(sample_graph: nx.Graph, tmp_path: pytest.TempPathFactory) -> None:
    """Test that node and edge features are preserved after serialization."""
    file_path = os.path.join(tmp_path, "test.pkl")
    serialization.serialize_graph(sample_graph, file_path)
    loaded_graph = serialization.deserialize_graph(file_path)
    assert loaded_graph.nodes[1]["feature"] == "A"
    assert loaded_graph.nodes[2]["feature"] == "B"
    assert loaded_graph.edges[1, 2]["weight"] == 1.5


def test_serialize_invalid_path(sample_graph: nx.Graph) -> None:
    """Test serializing to an invalid path raises an error."""
    with pytest.raises(OSError):
        serialization.serialize_graph(sample_graph, "/invalid_dir_path_that_does_not_exist_42/graph.pkl")


def test_deserialize_nonexistent_file() -> None:
    """Test deserializing from a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        serialization.deserialize_graph("nonexistent_file_path_12345.pkl")
