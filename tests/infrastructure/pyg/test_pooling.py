import pytest
import torch
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    TopKPooling,
    SAGPooling
)

def test_global_mean_pool_shape():
    """Test global_mean_pool output shape."""
    x = torch.randn(5, 16)
    batch = torch.tensor([0, 0, 1, 1, 2])
    out = global_mean_pool(x, batch)
    assert out.shape == (3, 16)

def test_global_mean_pool_values():
    """Test global_mean_pool output values."""
    x = torch.tensor([[1.0], [3.0], [4.0], [6.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = global_mean_pool(x, batch)
    assert out[0, 0].item() == 2.0
    assert out[1, 0].item() == 5.0

def test_global_mean_pool_batching():
    """Test global_mean_pool with unsorted batch indices."""
    x = torch.tensor([[1.0], [2.0], [3.0]])
    batch = torch.tensor([1, 0, 1])
    out = global_mean_pool(x, batch)
    assert out.shape == (2, 1)

def test_global_max_pool_shape():
    """Test global_max_pool output shape."""
    x = torch.randn(10, 8)
    batch = torch.tensor([0]*5 + [1]*5)
    out = global_max_pool(x, batch)
    assert out.shape == (2, 8)

def test_global_max_pool_values():
    """Test global_max_pool output values."""
    x = torch.tensor([[1.0], [5.0], [2.0], [8.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = global_max_pool(x, batch)
    assert out[0, 0].item() == 5.0
    assert out[1, 0].item() == 8.0

def test_global_max_pool_batching():
    """Test global_max_pool with varying batch sizes."""
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 1, 1, 2, 2, 2])
    out = global_max_pool(x, batch)
    assert out.shape == (3, 4)

def test_global_add_pool_shape():
    """Test global_add_pool output shape."""
    x = torch.randn(7, 32)
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    out = global_add_pool(x, batch)
    assert out.shape == (3, 32)

def test_global_add_pool_values():
    """Test global_add_pool output values."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = global_add_pool(x, batch)
    assert out[0, 0].item() == 3.0
    assert out[1, 0].item() == 7.0

def test_global_add_pool_batching():
    """Test global_add_pool missing batch indices."""
    x = torch.tensor([[1.0], [2.0]])
    batch = torch.tensor([0, 2])
    out = global_add_pool(x, batch, size=3)
    assert out.shape == (3, 1)
    assert out[1, 0].item() == 0.0

def test_topk_pooling_init():
    """Test TopKPooling initialization."""
    pool = TopKPooling(in_channels=16, ratio=0.5)
    assert pool.in_channels == 16
    assert pool.ratio == 0.5

def test_topk_pooling_forward():
    """Test TopKPooling forward pass."""
    pool = TopKPooling(in_channels=4, ratio=0.5)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out_x, out_edge_index, out_edge_attr, out_batch, perm, score = pool(x, edge_index, None, batch)
    assert out_x.shape[0] == 4  # ceil(3*0.5)*2
    assert out_batch.shape[0] == 4

def test_topk_pooling_ratio():
    """Test TopKPooling with ratio=1.0."""
    pool = TopKPooling(in_channels=2, ratio=1.0)
    x = torch.randn(4, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    out_x, _, _, _, _, _ = pool(x, edge_index)
    assert out_x.shape == (4, 2)

def test_sag_pooling_init():
    """Test SAGPooling initialization."""
    pool = SAGPooling(in_channels=16, ratio=0.8)
    assert pool.in_channels == 16
    assert pool.ratio == 0.8

def test_sag_pooling_forward():
    """Test SAGPooling forward pass."""
    pool = SAGPooling(in_channels=4, ratio=0.5)
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out_x, out_edge_index, out_edge_attr, out_batch, perm, score = pool(x, edge_index, None, batch)
    assert out_x.shape[0] == 4
    assert out_batch.shape[0] == 4

def test_sag_pooling_ratio():
    """Test SAGPooling with varied ratio."""
    pool = SAGPooling(in_channels=8, ratio=0.25)
    x = torch.randn(8, 8)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out_x, _, _, _, _, _ = pool(x, edge_index)
    assert out_x.shape[0] == 2
