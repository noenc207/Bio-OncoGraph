"""Tests for PyG batching."""
import pytest
import torch
from torch_geometric.data import Data, Batch

@pytest.fixture
def graphs_with_metadata():
    g1 = Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0]), graph_id=1, name="graph1")
    g2 = Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=torch.tensor([1]), graph_id=2, name="graph2")
    return [g1, g2]

def test_batch_creation(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert batch.num_graphs == 2

def test_batch_node_features(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert batch.x.shape == (7, 2)

def test_batch_edge_index(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert batch.edge_index.shape == (2, 5)
    # Check if edges are offset correctly
    assert torch.equal(batch.edge_index[:, 2:], torch.tensor([[0, 1, 2], [1, 2, 3]]) + 3)

def test_batch_labels(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert torch.equal(batch.y, torch.tensor([0, 1]))

def test_batch_metadata_lists(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert batch.graph_id == [1, 2]
    assert batch.name == ["graph1", "graph2"]

def test_batch_ptr(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert hasattr(batch, 'ptr')
    assert torch.equal(batch.ptr, torch.tensor([0, 3, 7]))

def test_batch_batch_vector(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    assert hasattr(batch, 'batch')
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1, 1, 1]))

def test_batch_to_data_list(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    data_list = batch.to_data_list()
    assert len(data_list) == 2
    assert data_list[0].x.shape == (3, 2)
    assert data_list[1].x.shape == (4, 2)

def test_batch_metadata_preservation_after_split(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    data_list = batch.to_data_list()
    assert data_list[0].graph_id == 1
    assert data_list[1].name == "graph2"

def test_batch_empty_list():
    with pytest.raises(Exception):
        Batch.from_data_list([])

def test_batch_single_graph(graphs_with_metadata):
    batch = Batch.from_data_list([graphs_with_metadata[0]])
    assert batch.num_graphs == 1
    assert batch.x.shape == (3, 2)

def test_batch_device_transfer(graphs_with_metadata):
    batch = Batch.from_data_list(graphs_with_metadata)
    batch = batch.to('cpu')
    assert batch.x.device.type == 'cpu'
