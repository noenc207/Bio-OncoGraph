"""Tests for PyG converters."""
import pytest
import torch
import torch_geometric
from torch_geometric.data import Data

def tensor_to_data(features, adjacency, labels=None):
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
    return Data(x=features, edge_index=edge_index, y=labels)

def dense_to_sparse_edge_index(dense_adj):
    return dense_adj.nonzero(as_tuple=False).t().contiguous()

@pytest.fixture
def sample_tensors():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adjacency = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    labels = torch.tensor([0, 1, 0])
    return features, adjacency, labels

def test_tensor_to_data_features(sample_tensors):
    features, adjacency, labels = sample_tensors
    data = tensor_to_data(features, adjacency)
    assert torch.equal(data.x, features)

def test_tensor_to_data_edge_index(sample_tensors):
    features, adjacency, labels = sample_tensors
    data = tensor_to_data(features, adjacency)
    expected_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    assert torch.equal(data.edge_index, expected_edge_index)

def test_tensor_to_data_labels(sample_tensors):
    features, adjacency, labels = sample_tensors
    data = tensor_to_data(features, adjacency, labels)
    assert torch.equal(data.y, labels)

def test_dense_to_sparse_basic():
    adj = torch.tensor([[0, 1], [1, 0]])
    edge_index = dense_to_sparse_edge_index(adj)
    assert torch.equal(edge_index, torch.tensor([[0, 1], [1, 0]]))

def test_dense_to_sparse_empty():
    adj = torch.zeros((3, 3), dtype=torch.long)
    edge_index = dense_to_sparse_edge_index(adj)
    assert edge_index.shape == (2, 0)

def test_dense_to_sparse_full():
    adj = torch.ones((2, 2), dtype=torch.long)
    edge_index = dense_to_sparse_edge_index(adj)
    assert edge_index.shape == (2, 4)

def test_tensor_to_data_no_edges():
    features = torch.ones((3, 2))
    adjacency = torch.zeros((3, 3))
    data = tensor_to_data(features, adjacency)
    assert data.edge_index.shape == (2, 0)

def test_tensor_to_data_self_loops():
    features = torch.ones((2, 2))
    adjacency = torch.tensor([[1, 0], [0, 1]])
    data = tensor_to_data(features, adjacency)
    assert torch.equal(data.edge_index, torch.tensor([[0, 1], [0, 1]]))

def test_tensor_to_data_dtype(sample_tensors):
    features, adjacency, labels = sample_tensors
    data = tensor_to_data(features, adjacency)
    assert data.edge_index.dtype == torch.long

def test_tensor_to_data_directed():
    features = torch.ones((3, 2))
    adjacency = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    data = tensor_to_data(features, adjacency)
    assert torch.equal(data.edge_index, torch.tensor([[0, 1], [1, 2]]))

def test_tensor_to_data_large_graph():
    features = torch.rand((100, 10))
    adjacency = torch.randint(0, 2, (100, 100))
    data = tensor_to_data(features, adjacency)
    assert data.x.shape == (100, 10)
    assert data.edge_index.shape[0] == 2

def test_sparse_tensor_support():
    # If using torch.sparse
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
    values = torch.tensor([1, 2, 3], dtype=torch.float)
    sparse_adj = torch.sparse_coo_tensor(indices, values, (3, 3))
    assert sparse_adj.is_sparse
