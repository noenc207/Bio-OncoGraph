"""Tests for PyG adapters."""
import pytest
import torch
import torch_geometric
from torch_geometric.data import Data, Batch

class DummyFrameworkGraph:
    def __init__(self, nodes, edges, node_attrs, edge_attrs, labels=None, masks=None, metadata=None):
        self.nodes = nodes
        self.edges = edges
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.labels = labels
        self.masks = masks
        self.metadata = metadata or {}

def convert_to_pyg(graph: DummyFrameworkGraph) -> Data:
    edge_index = torch.tensor(graph.edges, dtype=torch.long).t().contiguous() if graph.edges else torch.empty((2, 0), dtype=torch.long)
    return Data(
        x=torch.tensor(graph.node_attrs, dtype=torch.float) if graph.node_attrs is not None else None,
        edge_index=edge_index,
        edge_attr=torch.tensor(graph.edge_attrs, dtype=torch.float) if graph.edge_attrs is not None else None,
        y=torch.tensor(graph.labels, dtype=torch.long) if graph.labels is not None else None,
        mask=torch.tensor(graph.masks, dtype=torch.bool) if graph.masks is not None else None,
        **graph.metadata
    )

def convert_from_pyg(data: Data) -> DummyFrameworkGraph:
    edges = data.edge_index.t().tolist() if data.edge_index is not None and data.edge_index.numel() > 0 else []
    nodes = list(range(data.num_nodes)) if data.x is not None else []
    return DummyFrameworkGraph(
        nodes=nodes,
        edges=edges,
        node_attrs=data.x.tolist() if data.x is not None else None,
        edge_attrs=data.edge_attr.tolist() if data.edge_attr is not None else None,
        labels=data.y.tolist() if data.y is not None else None,
        masks=data.mask.tolist() if hasattr(data, 'mask') and data.mask is not None else None
    )

@pytest.fixture
def sample_graph():
    return DummyFrameworkGraph(
        nodes=[0, 1, 2],
        edges=[[0, 1], [1, 2]],
        node_attrs=[[1.0], [2.0], [3.0]],
        edge_attrs=[[0.5], [0.6]],
        labels=[0, 1, 0],
        masks=[True, False, True],
        metadata={'graph_id': 1}
    )

def test_adapter_to_pyg_nodes(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert data.x.shape == (3, 1)
    assert torch.allclose(data.x, torch.tensor([[1.0], [2.0], [3.0]]))

def test_adapter_to_pyg_edges(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert data.edge_index.shape == (2, 2)
    assert data.edge_index.tolist() == [[0, 1], [1, 2]]

def test_adapter_to_pyg_edge_attrs(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert data.edge_attr.shape == (2, 1)

def test_adapter_to_pyg_labels(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert data.y.tolist() == [0, 1, 0]

def test_adapter_to_pyg_masks(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert data.mask.tolist() == [True, False, True]

def test_adapter_to_pyg_metadata(sample_graph):
    data = convert_to_pyg(sample_graph)
    assert getattr(data, 'graph_id') == 1

def test_adapter_from_pyg_nodes(sample_graph):
    data = convert_to_pyg(sample_graph)
    graph = convert_from_pyg(data)
    assert len(graph.nodes) == 3

def test_adapter_from_pyg_edges(sample_graph):
    data = convert_to_pyg(sample_graph)
    graph = convert_from_pyg(data)
    assert graph.edges == [[0, 1], [1, 2]]

def test_adapter_from_pyg_labels(sample_graph):
    data = convert_to_pyg(sample_graph)
    graph = convert_from_pyg(data)
    assert graph.labels == [0, 1, 0]

def test_adapter_from_pyg_masks(sample_graph):
    data = convert_to_pyg(sample_graph)
    graph = convert_from_pyg(data)
    assert graph.masks == [True, False, True]

def test_adapter_empty_graph():
    empty_graph = DummyFrameworkGraph([], [], None, None)
    data = convert_to_pyg(empty_graph)
    assert data.x is None
    assert data.edge_index.numel() == 0
    back_graph = convert_from_pyg(data)
    assert back_graph.nodes == []
    assert back_graph.edges == []

def test_adapter_batching(sample_graph):
    data1 = convert_to_pyg(sample_graph)
    data2 = convert_to_pyg(sample_graph)
    batch = Batch.from_data_list([data1, data2])
    assert batch.x.shape == (6, 1)
    assert batch.edge_index.shape == (2, 4)
