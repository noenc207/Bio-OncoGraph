"""Tests for GraphSerializer and HierarchySerializer."""

import pytest
import json
import os
from unittest.mock import MagicMock

class GraphSerializer:
    def save(self, graph, filepath):
        with open(filepath, 'w') as f:
            json.dump({'nodes': graph['nodes'], 'edges': graph['edges'], 'version': '1.0'}, f)
    def load(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

class HierarchySerializer:
    def save(self, hierarchy, filepath):
        with open(filepath, 'w') as f:
            json.dump({'levels': hierarchy['levels'], 'version': '1.1'}, f)
    def load(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

@pytest.fixture
def graph_serializer():
    return GraphSerializer()

@pytest.fixture
def hierarchy_serializer():
    return HierarchySerializer()

@pytest.fixture
def sample_graph():
    return {'nodes': [1, 2], 'edges': [[1, 2]]}

@pytest.fixture
def sample_hierarchy():
    return {'levels': ['cell', 'tissue']}

def test_graph_serializer_save(graph_serializer, sample_graph, tmp_path):
    filepath = tmp_path / "graph.json"
    graph_serializer.save(sample_graph, filepath)
    assert filepath.exists()

def test_graph_serializer_load(graph_serializer, sample_graph, tmp_path):
    filepath = tmp_path / "graph.json"
    graph_serializer.save(sample_graph, filepath)
    loaded = graph_serializer.load(filepath)
    assert loaded['nodes'] == sample_graph['nodes']
    assert loaded['edges'] == sample_graph['edges']

def test_graph_serializer_version(graph_serializer, sample_graph, tmp_path):
    filepath = tmp_path / "graph.json"
    graph_serializer.save(sample_graph, filepath)
    loaded = graph_serializer.load(filepath)
    assert loaded['version'] == '1.0'

def test_hierarchy_serializer_save(hierarchy_serializer, sample_hierarchy, tmp_path):
    filepath = tmp_path / "hierarchy.json"
    hierarchy_serializer.save(sample_hierarchy, filepath)
    assert filepath.exists()

def test_hierarchy_serializer_load(hierarchy_serializer, sample_hierarchy, tmp_path):
    filepath = tmp_path / "hierarchy.json"
    hierarchy_serializer.save(sample_hierarchy, filepath)
    loaded = hierarchy_serializer.load(filepath)
    assert loaded['levels'] == sample_hierarchy['levels']

def test_hierarchy_serializer_version(hierarchy_serializer, sample_hierarchy, tmp_path):
    filepath = tmp_path / "hierarchy.json"
    hierarchy_serializer.save(sample_hierarchy, filepath)
    loaded = hierarchy_serializer.load(filepath)
    assert loaded['version'] == '1.1'

def test_graph_serializer_empty(graph_serializer, tmp_path):
    filepath = tmp_path / "empty.json"
    graph_serializer.save({'nodes': [], 'edges': []}, filepath)
    loaded = graph_serializer.load(filepath)
    assert loaded['nodes'] == []

def test_hierarchy_serializer_empty(hierarchy_serializer, tmp_path):
    filepath = tmp_path / "empty_hier.json"
    hierarchy_serializer.save({'levels': []}, filepath)
    loaded = hierarchy_serializer.load(filepath)
    assert loaded['levels'] == []

def test_graph_serializer_invalid_path(graph_serializer):
    with pytest.raises(FileNotFoundError):
        graph_serializer.load("non_existent_file.json")

def test_hierarchy_serializer_invalid_path(hierarchy_serializer):
    with pytest.raises(FileNotFoundError):
        hierarchy_serializer.load("non_existent_file.json")

def test_graph_serializer_large_data(graph_serializer, tmp_path):
    filepath = tmp_path / "large_graph.json"
    large_graph = {'nodes': list(range(1000)), 'edges': []}
    graph_serializer.save(large_graph, filepath)
    loaded = graph_serializer.load(filepath)
    assert len(loaded['nodes']) == 1000

def test_hierarchy_serializer_many_levels(hierarchy_serializer, tmp_path):
    filepath = tmp_path / "large_hier.json"
    large_hier = {'levels': [f'level_{i}' for i in range(100)]}
    hierarchy_serializer.save(large_hier, filepath)
    loaded = hierarchy_serializer.load(filepath)
    assert len(loaded['levels']) == 100

def test_graph_serializer_roundtrip_fidelity(graph_serializer, sample_graph, tmp_path):
    filepath = tmp_path / "rt.json"
    graph_serializer.save(sample_graph, filepath)
    loaded = graph_serializer.load(filepath)
    assert loaded == {'nodes': [1, 2], 'edges': [[1, 2]], 'version': '1.0'}
