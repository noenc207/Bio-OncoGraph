"""Tests for PyG utilities."""
import pytest
import torch
import torch_geometric
from torch_geometric.utils import add_self_loops, remove_self_loops, degree, to_dense_adj

@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

def test_add_self_loops(edge_index):
    new_edge_index, _ = add_self_loops(edge_index, num_nodes=3)
    assert new_edge_index.shape[1] == 7 # 4 original + 3 loops

def test_remove_self_loops():
    ei = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long)
    new_ei, _ = remove_self_loops(ei)
    assert new_ei.shape[1] == 1
    assert torch.equal(new_ei, torch.tensor([[1], [2]]))

def test_degree(edge_index):
    row, col = edge_index
    deg = degree(row, num_nodes=3)
    assert torch.equal(deg, torch.tensor([1.0, 2.0, 1.0]))

def test_to_dense_adj(edge_index):
    adj = to_dense_adj(edge_index, max_num_nodes=3)
    assert adj.shape == (1, 3, 3)
    assert adj[0, 0, 1] == 1
    assert adj[0, 1, 0] == 1

def test_contains_self_loops(edge_index):
    from torch_geometric.utils import contains_self_loops
    assert not contains_self_loops(edge_index)

def test_contains_isolated_nodes(edge_index):
    from torch_geometric.utils import contains_isolated_nodes
    assert not contains_isolated_nodes(edge_index, num_nodes=3)
    assert contains_isolated_nodes(edge_index, num_nodes=4)

def test_is_undirected(edge_index):
    from torch_geometric.utils import is_undirected
    assert is_undirected(edge_index)

def test_to_undirected():
    from torch_geometric.utils import to_undirected
    ei = torch.tensor([[0, 1], [1, 2]])
    undirected_ei = to_undirected(ei)
    assert undirected_ei.shape[1] == 4

def test_subgraph(edge_index):
    from torch_geometric.utils import subgraph
    subset = torch.tensor([0, 1])
    sub_ei, _ = subgraph(subset, edge_index, num_nodes=3)
    assert sub_ei.shape[1] == 2
    assert torch.equal(sub_ei, torch.tensor([[0, 1], [1, 0]]))

def test_k_hop_subgraph(edge_index):
    from torch_geometric.utils import k_hop_subgraph
    subset, sub_ei, mapping, edge_mask = k_hop_subgraph(0, 1, edge_index, num_nodes=3)
    assert torch.equal(subset, torch.tensor([0, 1]))

def test_negative_sampling(edge_index):
    from torch_geometric.utils import negative_sampling
    neg_ei = negative_sampling(edge_index, num_nodes=3, num_neg_samples=2)
    assert neg_ei.shape[1] == 2

def test_batched_negative_sampling():
    from torch_geometric.utils import batched_negative_sampling
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0, 0, 1, 1])
    neg_ei = batched_negative_sampling(ei, batch)
    assert neg_ei.shape[0] == 2
