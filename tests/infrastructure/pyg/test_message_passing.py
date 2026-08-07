import pytest
import torch
from torch_geometric.nn import MessagePassing

class MockGNN(MessagePassing):
    """Mock Graph Neural Network layer for testing MessagePassing wrapper."""
    def __init__(self, aggr="add", **kwargs):
        super().__init__(aggr=aggr, **kwargs)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j + edge_attr
        return x_j

class MockUpdateGNN(MessagePassing):
    """Mock GNN layer with overridden update function."""
    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out, x):
        return aggr_out + x


def test_message_passing_init():
    """Test initialization of MessagePassing wrapper."""
    gnn = MockGNN(aggr="mean")
    assert gnn.aggr == "mean"

def test_message_passing_forward():
    """Test standard forward pass aggregation."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    out = gnn(x, edge_index)
    assert out.shape == (3, 1)

def test_message_passing_aggr_add():
    """Test additive aggregation."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[1.0], [1.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    out = gnn(x, edge_index)
    assert out[2].item() == 2.0

def test_message_passing_aggr_mean():
    """Test mean aggregation."""
    gnn = MockGNN(aggr="mean")
    x = torch.tensor([[2.0], [4.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    out = gnn(x, edge_index)
    assert out[2].item() == 3.0

def test_message_passing_aggr_max():
    """Test max aggregation."""
    gnn = MockGNN(aggr="max")
    x = torch.tensor([[2.0], [4.0], [1.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    out = gnn(x, edge_index)
    assert out[2].item() == 4.0

def test_message_passing_edge_attributes():
    """Test message passing with edge attributes."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.tensor([[0.5], [1.5]])
    out = gnn(x, edge_index, edge_attr)
    assert out[1].item() == 1.5
    assert out[2].item() == 3.5

def test_message_passing_update_override():
    """Test GNN with overridden update function."""
    gnn = MockUpdateGNN()
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    out = gnn(x, edge_index)
    assert out[1].item() == 3.0
    assert out[2].item() == 5.0

def test_message_passing_propagate_empty_graph():
    """Test propagation on an empty graph."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = gnn(x, edge_index)
    assert out.shape == (2, 1)
    assert torch.all(out == 0)

def test_message_passing_propagate_single_node():
    """Test propagation on a single node graph."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[5.0]])
    edge_index = torch.tensor([[0], [0]])
    out = gnn(x, edge_index)
    assert out[0].item() == 5.0

def test_message_passing_message_and_aggregate_override():
    """Test internal message and aggregate logic."""
    class CustomAggrGNN(MockGNN):
        def aggregate(self, inputs, index, ptr=None, dim_size=None):
            return super().aggregate(inputs * 2, index, ptr, dim_size)
            
    gnn = CustomAggrGNN(aggr="add")
    x = torch.tensor([[1.0], [2.0]])
    edge_index = torch.tensor([[0], [1]])
    out = gnn(x, edge_index)
    assert out[1].item() == 2.0

def test_message_passing_bipartite_graph():
    """Test message passing on bipartite graphs."""
    gnn = MockGNN(aggr="add")
    x = (torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]]))
    edge_index = torch.tensor([[0, 1], [0, 1]])
    out = gnn(x, edge_index)
    assert out.shape == (2, 1)

def test_message_passing_self_loops():
    """Test message passing with self-loops."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[2.0]])
    edge_index = torch.tensor([[0, 0, 0], [0, 0, 0]])
    out = gnn(x, edge_index)
    assert out[0].item() == 6.0

def test_message_passing_disconnected_graph():
    """Test message passing on completely disconnected graph."""
    gnn = MockGNN(aggr="add")
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.empty((2, 0), dtype=torch.long)
    out = gnn(x, edge_index)
    assert torch.all(out == 0)
