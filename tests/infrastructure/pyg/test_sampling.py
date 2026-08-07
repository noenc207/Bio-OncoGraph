import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, RandomNodeSampler, ImbalancedSampler

# Dummy graph for sampling tests
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],
    [1, 0, 2, 1, 3, 2, 4, 3]
])
x = torch.randn(5, 16)
data = Data(x=x, edge_index=edge_index)

def test_neighbor_sampler_init():
    """Test NeighborLoader initialization."""
    loader = NeighborLoader(data, num_neighbors=[2, 2], batch_size=2)
    assert loader.batch_size == 2

def test_neighbor_sampler_sample():
    """Test NeighborLoader sampling output."""
    loader = NeighborLoader(data, num_neighbors=[2], batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, Data)
    assert batch.batch_size == 2

def test_neighbor_sampler_sizes():
    """Test NeighborLoader sizes logic."""
    loader = NeighborLoader(data, num_neighbors=[1, 1], batch_size=1)
    batch = next(iter(loader))
    assert batch.x.shape[0] <= 3  # root + 1 neighbor + 1 neighbor of neighbor

def test_neighbor_sampler_batch_size():
    """Test NeighborLoader full batch loading."""
    loader = NeighborLoader(data, num_neighbors=[2, 2], batch_size=5)
    batch = next(iter(loader))
    assert batch.x.shape[0] == 5

def test_neighbor_sampler_directed():
    """Test NeighborLoader with directed edges."""
    dir_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    dir_data = Data(x=torch.randn(4, 4), edge_index=dir_edge_index)
    loader = NeighborLoader(dir_data, num_neighbors=[1], batch_size=1, directed=True)
    batch = next(iter(loader))
    assert batch.x.shape[0] <= 2

def test_subgraph_sampler_init():
    """Test RandomNodeSampler initialization."""
    loader = RandomNodeSampler(data, num_parts=2)
    assert loader.num_parts == 2

def test_subgraph_sampler_nodes():
    """Test RandomNodeSampler node count."""
    loader = RandomNodeSampler(data, num_parts=2)
    batch = next(iter(loader))
    assert batch.num_nodes > 0
    assert batch.num_nodes <= 5

def test_subgraph_sampler_edges():
    """Test RandomNodeSampler edge index format."""
    loader = RandomNodeSampler(data, num_parts=2)
    batch = next(iter(loader))
    assert batch.edge_index.shape[0] == 2

def test_random_walk_init():
    """Test ImbalancedSampler initialization."""
    y = torch.tensor([0, 0, 1, 1, 1])
    data.y = y
    sampler = ImbalancedSampler(data)
    assert sampler is not None

def test_random_walk_length():
    """Test ImbalancedSampler length matches dataset."""
    y = torch.tensor([0, 0, 1, 1, 1])
    data.y = y
    sampler = ImbalancedSampler(data)
    assert len(list(sampler)) == 5

def test_random_walk_start_nodes():
    """Test sampling start nodes logic."""
    y = torch.tensor([0, 1])
    temp_data = Data(x=torch.randn(2, 2), edge_index=torch.tensor([[0, 1], [1, 0]]), y=y)
    sampler = ImbalancedSampler(temp_data)
    idx = list(sampler)
    assert len(idx) == 2

def test_random_walk_bipartite():
    """Test sampling mock for bipartite."""
    bipartite_data = Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1], [2, 3]]))
    loader = NeighborLoader(bipartite_data, num_neighbors=[1], batch_size=2)
    batch = next(iter(loader))
    assert batch is not None

def test_neighbor_sampler_empty():
    """Test NeighborLoader with empty graph."""
    empty_data = Data(x=torch.randn(0, 4), edge_index=torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(Exception):
        loader = NeighborLoader(empty_data, num_neighbors=[1], batch_size=2)
        next(iter(loader))
