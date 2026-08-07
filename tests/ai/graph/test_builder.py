"""Tests for the GraphBuilder class."""

import os
from typing import Any

import networkx as nx
import pytest

from bio_oncograph.ai.graph.builder import GraphBuilder
from bio_oncograph.ai.graph.report import GraphReport


def test_builder_creation() -> None:
    """Test creation of GraphBuilder with and without config."""
    builder_default = GraphBuilder()
    assert builder_default is not None

    builder_with_config = GraphBuilder(config={"mode": "fast"})
    assert builder_with_config is not None


def test_builder_build() -> None:
    """Test the build method of GraphBuilder."""
    builder = GraphBuilder()
    graph = builder.build()
    assert graph is not None


def test_builder_validate() -> None:
    """Test the validate method of GraphBuilder."""
    builder = GraphBuilder()
    graph = builder.build()
    is_valid = builder.validate(graph)
    assert isinstance(is_valid, bool)


def test_builder_statistics() -> None:
    """Test the statistics method of GraphBuilder."""
    builder = GraphBuilder()
    graph = builder.build()
    stats = builder.statistics(graph)
    assert isinstance(stats, dict)
    assert "nodes" in stats


def test_builder_serialize_deserialize(tmp_path: pytest.TempPathFactory) -> None:
    """Test serialize and deserialize methods of GraphBuilder."""
    builder = GraphBuilder()
    graph = builder.build()
    file_path = os.path.join(tmp_path, "builder_graph.pkl")
    
    builder.serialize(graph, file_path)
    assert os.path.exists(file_path)
    
    loaded_graph = builder.deserialize(file_path)
    assert loaded_graph is not None


def test_builder_report() -> None:
    """Test the report method of GraphBuilder."""
    builder = GraphBuilder()
    graph = builder.build()
    report = builder.report(graph)
    assert isinstance(report, GraphReport)


def test_builder_invalid_graph_validation_returns_false() -> None:
    """Test that validating an invalid graph returns False."""
    builder = GraphBuilder()
    # Assuming an empty list is an invalid graph representation for the builder
    invalid_graph: Any = []
    is_valid = builder.validate(invalid_graph)
    assert is_valid is False


def test_builder_build_with_custom_parameters() -> None:
    """Test building a graph with custom parameters."""
    builder = GraphBuilder(config={"node_limit": 100})
    graph = builder.build(node_limit=50)
    assert graph is not None
