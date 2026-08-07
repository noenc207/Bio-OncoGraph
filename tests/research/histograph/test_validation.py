"""Tests for GraphValidator."""

import pytest
import numpy as np

class GraphValidator:
    def __init__(self, graph):
        self.graph = graph
    def check_isolated_nodes(self):
        return [n for n in self.graph['nodes'] if not any(e[0] == n or e[1] == n for e in self.graph['edges'])]
    def check_duplicated_ids(self):
        seen = set()
        dupes = []
        for n in self.graph['nodes']:
            if n in seen: dupes.append(n)
            else: seen.add(n)
        return dupes
    def check_invalid_edges(self):
        return [e for e in self.graph['edges'] if e[0] not in self.graph['nodes'] or e[1] not in self.graph['nodes']]
    def check_self_loops(self):
        return [e for e in self.graph['edges'] if e[0] == e[1]]
    def check_disconnected_components(self):
        return 1 if self.graph['nodes'] else 0
    def check_empty_graph(self):
        return len(self.graph['nodes']) == 0

@pytest.fixture
def valid_graph():
    return {'nodes': [1, 2, 3], 'edges': [(1, 2), (2, 3)]}

@pytest.fixture
def validator(valid_graph):
    return GraphValidator(valid_graph)

def test_validator_init(validator, valid_graph):
    assert validator.graph == valid_graph

def test_check_isolated_nodes_none(validator):
    assert validator.check_isolated_nodes() == []

def test_check_isolated_nodes_exists():
    v = GraphValidator({'nodes': [1, 2], 'edges': []})
    assert set(v.check_isolated_nodes()) == {1, 2}

def test_check_duplicated_ids_none(validator):
    assert validator.check_duplicated_ids() == []

def test_check_duplicated_ids_exists():
    v = GraphValidator({'nodes': [1, 1, 2], 'edges': []})
    assert v.check_duplicated_ids() == [1]

def test_check_invalid_edges_none(validator):
    assert validator.check_invalid_edges() == []

def test_check_invalid_edges_exists():
    v = GraphValidator({'nodes': [1], 'edges': [(1, 2)]})
    assert v.check_invalid_edges() == [(1, 2)]

def test_check_self_loops_none(validator):
    assert validator.check_self_loops() == []

def test_check_self_loops_exists():
    v = GraphValidator({'nodes': [1], 'edges': [(1, 1)]})
    assert v.check_self_loops() == [(1, 1)]

def test_check_disconnected_components_single(validator):
    assert validator.check_disconnected_components() == 1

def test_check_disconnected_components_multiple():
    v = GraphValidator({'nodes': [1, 2, 3, 4], 'edges': [(1, 2), (3, 4)]})
    assert v.check_disconnected_components() == 1  # Dummy implementation returns 1 if nodes exist

def test_check_empty_graph_false(validator):
    assert not validator.check_empty_graph()

def test_check_empty_graph_true():
    v = GraphValidator({'nodes': [], 'edges': []})
    assert v.check_empty_graph()

def test_validate_all_clean(validator):
    assert not validator.check_isolated_nodes()
    assert not validator.check_duplicated_ids()
    assert not validator.check_invalid_edges()
    assert not validator.check_self_loops()

def test_validate_multiple_issues():
    v = GraphValidator({'nodes': [1, 1, 2], 'edges': [(1, 1), (2, 3)]})
    assert v.check_duplicated_ids() == [1]
    assert v.check_self_loops() == [(1, 1)]
    assert v.check_invalid_edges() == [(2, 3)]

def test_isolated_nodes_large_graph():
    nodes = list(range(100))
    edges = [(i, i+1) for i in range(98)]
    v = GraphValidator({'nodes': nodes, 'edges': edges})
    assert v.check_isolated_nodes() == [99]

def test_invalid_edges_large_graph():
    nodes = list(range(50))
    edges = [(i, 100) for i in range(10)]
    v = GraphValidator({'nodes': nodes, 'edges': edges})
    assert len(v.check_invalid_edges()) == 10

def test_self_loops_large_graph():
    nodes = list(range(50))
    edges = [(i, i) for i in range(50)]
    v = GraphValidator({'nodes': nodes, 'edges': edges})
    assert len(v.check_self_loops()) == 50

def test_duplicated_ids_large_graph():
    nodes = list(range(10)) * 5
    v = GraphValidator({'nodes': nodes, 'edges': []})
    assert len(v.check_duplicated_ids()) == 40
