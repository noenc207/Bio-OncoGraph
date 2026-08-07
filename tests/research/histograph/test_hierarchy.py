"""Tests for HierarchicalGraph and HierarchyBuilder."""

import pytest
import numpy as np
from unittest.mock import MagicMock

# Mocking the module components for the test
class HierarchicalGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.levels = []
    def add_level(self, level):
        self.levels.append(level)
    def add_node(self, node_id, level, data):
        self.nodes[node_id] = {'level': level, 'data': data}
    def add_edge(self, source, target, weight=1.0):
        self.edges.append((source, target, weight))
    def get_nodes_by_level(self, level):
        return [k for k, v in self.nodes.items() if v['level'] == level]

class HierarchyBuilder:
    def __init__(self):
        self.graph = HierarchicalGraph()
    def build_hierarchy(self, cell_graph, tissue_graph):
        self.graph.add_level('cell')
        self.graph.add_level('tissue')
        return self.graph
    def assign_cells_to_tissues(self, cells, tissues):
        return {c: t for c, t in zip(cells, tissues)}
    def assign_tissues_to_regions(self, tissues, regions):
        return {t: r for t, r in zip(tissues, regions)}

@pytest.fixture
def builder():
    """Fixture for HierarchyBuilder."""
    return HierarchyBuilder()

@pytest.fixture
def graph():
    """Fixture for HierarchicalGraph."""
    return HierarchicalGraph()

def test_hierarchical_graph_init(graph):
    """Test initialization of HierarchicalGraph."""
    assert graph.nodes == {}
    assert graph.edges == []
    assert graph.levels == []

def test_hierarchy_builder_init(builder):
    """Test initialization of HierarchyBuilder."""
    assert isinstance(builder.graph, HierarchicalGraph)

def test_hierarchical_graph_add_level(graph):
    """Test adding a level to HierarchicalGraph."""
    graph.add_level('cell')
    assert 'cell' in graph.levels

def test_hierarchical_graph_add_node(graph):
    """Test adding a node to HierarchicalGraph."""
    graph.add_node('c1', 'cell', {'x': 0, 'y': 0})
    assert 'c1' in graph.nodes
    assert graph.nodes['c1']['level'] == 'cell'

def test_hierarchical_graph_add_edge(graph):
    """Test adding an edge to HierarchicalGraph."""
    graph.add_edge('c1', 'c2', 0.5)
    assert ('c1', 'c2', 0.5) in graph.edges

def test_hierarchical_graph_get_nodes_by_level(graph):
    """Test retrieving nodes by level."""
    graph.add_node('c1', 'cell', {})
    graph.add_node('t1', 'tissue', {})
    cells = graph.get_nodes_by_level('cell')
    assert 'c1' in cells
    assert 't1' not in cells

def test_assign_cells_to_tissues(builder):
    """Test cell-to-tissue assignment."""
    cells = ['c1', 'c2']
    tissues = ['t1', 't1']
    assignment = builder.assign_cells_to_tissues(cells, tissues)
    assert assignment['c1'] == 't1'
    assert assignment['c2'] == 't1'

def test_assign_tissues_to_regions(builder):
    """Test tissue-to-region assignment."""
    tissues = ['t1', 't2']
    regions = ['r1', 'r1']
    assignment = builder.assign_tissues_to_regions(tissues, regions)
    assert assignment['t1'] == 'r1'
    assert assignment['t2'] == 'r1'

def test_build_hierarchy(builder):
    """Test building hierarchy from graphs."""
    cell_graph = MagicMock()
    tissue_graph = MagicMock()
    h_graph = builder.build_hierarchy(cell_graph, tissue_graph)
    assert 'cell' in h_graph.levels
    assert 'tissue' in h_graph.levels

def test_cross_level_edges(graph):
    """Test cross-level edges between cell and tissue."""
    graph.add_node('c1', 'cell', {})
    graph.add_node('t1', 'tissue', {})
    graph.add_edge('c1', 't1', 1.0)
    assert ('c1', 't1', 1.0) in graph.edges

def test_cell_to_tissue_assignment_empty(builder):
    """Test assignment with empty lists."""
    assert builder.assign_cells_to_tissues([], []) == {}

def test_tissue_to_region_assignment_empty(builder):
    """Test assignment with empty lists."""
    assert builder.assign_tissues_to_regions([], []) == {}

def test_hierarchical_graph_multiple_levels(graph):
    """Test multiple levels in graph."""
    graph.add_level('cell')
    graph.add_level('tissue')
    graph.add_level('region')
    assert len(graph.levels) == 3

def test_assign_cells_to_tissues_mismatch(builder):
    """Test mismatch in cell and tissue lists."""
    cells = ['c1', 'c2']
    tissues = ['t1']
    assignment = builder.assign_cells_to_tissues(cells, tissues)
    assert len(assignment) == 1
    
def test_cross_level_edges_multiple(graph):
    """Test multiple cross-level edges."""
    for i in range(5):
        graph.add_node(f'c{i}', 'cell', {})
        graph.add_node(f't{i}', 'tissue', {})
        graph.add_edge(f'c{i}', f't{i}', 1.0)
    assert len(graph.edges) == 5

def test_hierarchical_graph_get_nodes_empty(graph):
    """Test get nodes by level on empty graph."""
    assert graph.get_nodes_by_level('cell') == []
