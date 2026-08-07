import pytest
import torch
from torch_geometric.data import Data

def validate_data(data: Data):
    """Mock validation function mimicking framework validators."""
    if not isinstance(data, Data):
        raise ValueError("Invalid graph type.")
    if data.edge_index is not None and data.edge_index.dim() != 2:
        raise ValueError("edge_index must be 2D.")
    if data.edge_index is not None and data.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if data.edge_index is not None and not data.edge_index.is_floating_point() and data.edge_index.dtype != torch.long:
        raise ValueError("edge_index dtype must be torch.long.")
    if data.x is None:
        raise ValueError("Missing node features.")
    if data.edge_index is not None and data.edge_index.numel() > 0 and data.x is not None:
        max_idx = data.edge_index.max().item()
        if max_idx >= data.x.shape[0]:
            raise ValueError("Feature shape mismatch with edge_index.")
    if hasattr(data, "meta"):
        if not isinstance(data.meta, dict):
            raise ValueError("Corrupted metadata: wrong type.")
        if "version" not in data.meta:
            raise ValueError("Corrupted metadata: missing key.")
    return True

def check_disconnected(data: Data):
    if data.edge_index is None or data.edge_index.numel() == 0:
        return True
    return False

def check_isolated_nodes(data: Data):
    if data.edge_index is None or data.x is None:
        return False
    unique_nodes = torch.unique(data.edge_index)
    return unique_nodes.numel() < data.x.shape[0]


def test_validate_valid_graph():
    """Test validation with a fully valid graph."""
    data = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1], [1, 2]]))
    assert validate_data(data)

def test_validate_invalid_graph_type():
    """Test validation with invalid type."""
    with pytest.raises(ValueError, match="Invalid graph type."):
        validate_data(dict(x=torch.randn(3, 4)))

def test_validate_edge_index_shape_1d():
    """Test validation with 1D edge_index."""
    data = Data(x=torch.randn(3, 4), edge_index=torch.tensor([0, 1, 2]))
    with pytest.raises(ValueError, match="edge_index must be 2D."):
        validate_data(data)

def test_validate_edge_index_shape_3d():
    """Test validation with 3D edge_index."""
    data = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[[0, 1]]]))
    with pytest.raises(ValueError, match="edge_index must be 2D."):
        validate_data(data)

def test_validate_edge_index_type_float():
    """Test validation with float edge_index."""
    data = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0.0, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="edge_index dtype must be torch.long."):
        validate_data(data)

def test_validate_missing_node_features():
    """Test validation with missing x."""
    data = Data(edge_index=torch.tensor([[0, 1], [1, 2]]))
    with pytest.raises(ValueError, match="Missing node features."):
        validate_data(data)

def test_validate_feature_shape_mismatch():
    """Test validation where edge_index points out of bounds."""
    data = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [1, 3]]))
    with pytest.raises(ValueError, match="Feature shape mismatch with edge_index."):
        validate_data(data)

def test_validate_disconnected_graph_warning():
    """Test detection of disconnected graph."""
    data = Data(x=torch.randn(2, 4), edge_index=torch.empty((2, 0), dtype=torch.long))
    assert check_disconnected(data)

def test_validate_disconnected_graph_error():
    """Test disconnected logic on connected graph."""
    data = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0], [1]]))
    assert not check_disconnected(data)

def test_validate_corrupted_metadata_missing_key():
    """Test metadata missing required key."""
    data = Data(x=torch.randn(2, 4), meta={"author": "test"})
    with pytest.raises(ValueError, match="Corrupted metadata: missing key."):
        validate_data(data)

def test_validate_corrupted_metadata_wrong_type():
    """Test metadata wrong type."""
    data = Data(x=torch.randn(2, 4), meta="invalid_meta")
    with pytest.raises(ValueError, match="Corrupted metadata: wrong type."):
        validate_data(data)

def test_validate_self_loops_present():
    """Test graph with self loops is valid."""
    data = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0, 1], [0, 1]]))
    assert validate_data(data)

def test_validate_isolated_nodes():
    """Test detection of isolated nodes."""
    data = Data(x=torch.randn(4, 4), edge_index=torch.tensor([[0, 1], [1, 0]]))
    assert check_isolated_nodes(data)
