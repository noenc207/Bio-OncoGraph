"""Tests for PyG dataloaders."""
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader

@pytest.fixture
def sample_data_list():
    return [
        Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0])),
        Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=torch.tensor([1])),
        Data(x=torch.randn(2, 2), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([0])),
        Data(x=torch.randn(5, 2), edge_index=torch.tensor([[0, 1, 3], [2, 4, 1]]), y=torch.tensor([1]))
    ]

@pytest.fixture
def large_graph():
    return Data(
        x=torch.randn(100, 5),
        edge_index=torch.randint(0, 100, (2, 300)),
        y=torch.randint(0, 2, (100,))
    )

def test_dataloader_basic(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2)
    assert len(loader) == 2

def test_dataloader_batch_size(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2)
    batch = next(iter(loader))
    assert batch.num_graphs == 2
    assert batch.x.shape == (7, 2)

def test_dataloader_shuffle(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2, shuffle=True)
    batches = list(loader)
    assert len(batches) == 2

def test_dataloader_drop_last(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=3, drop_last=True)
    assert len(loader) == 1
    batch = next(iter(loader))
    assert batch.num_graphs == 3

def test_neighbor_loader_basic(large_graph):
    loader = NeighborLoader(
        large_graph,
        num_neighbors=[10, 10],
        batch_size=16,
        input_nodes=None
    )
    batch = next(iter(loader))
    assert batch.batch_size == 16
    assert batch.x.shape[0] <= 100

def test_neighbor_loader_directed(large_graph):
    loader = NeighborLoader(
        large_graph,
        num_neighbors=[5],
        batch_size=8,
        directed=True
    )
    batch = next(iter(loader))
    assert batch.batch_size == 8

def test_neighbor_loader_epochs(large_graph):
    loader = NeighborLoader(large_graph, num_neighbors=[2], batch_size=32)
    count = sum(1 for _ in loader)
    expected = (100 + 32 - 1) // 32
    assert count == expected

def test_dataloader_pin_memory(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2, pin_memory=True)
    batch = next(iter(loader))
    if torch.cuda.is_available():
        assert batch.x.is_pinned()

def test_dataloader_num_workers(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2, num_workers=0)
    assert len(list(loader)) == 2

def test_neighbor_loader_input_nodes(large_graph):
    input_nodes = torch.arange(10)
    loader = NeighborLoader(large_graph, num_neighbors=[5], batch_size=5, input_nodes=input_nodes)
    assert len(loader) == 2

def test_dataloader_iteration_determinism(sample_data_list):
    loader = DataLoader(sample_data_list, batch_size=2, shuffle=False)
    b1 = next(iter(loader))
    b2 = next(iter(loader))
    assert torch.equal(b1.x, b2.x)

def test_neighbor_loader_replace(large_graph):
    loader = NeighborLoader(large_graph, num_neighbors=[10], batch_size=10, replace=True)
    batch = next(iter(loader))
    assert batch is not None
