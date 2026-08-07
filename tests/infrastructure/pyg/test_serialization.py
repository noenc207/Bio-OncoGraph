import pytest
import torch
import os
from torch_geometric.data import Data, Batch

def test_serialize_data_basic(tmp_path):
    """Test basic serialization of PyG Data."""
    file_path = os.path.join(tmp_path, "data.pt")
    data = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
    torch.save(data, file_path)
    assert os.path.exists(file_path)

def test_deserialize_data_basic(tmp_path):
    """Test basic deserialization of PyG Data."""
    file_path = os.path.join(tmp_path, "data.pt")
    data = Data(x=torch.tensor([[1.0]]), edge_index=torch.tensor([[0], [0]]))
    torch.save(data, file_path)
    loaded_data = torch.load(file_path)
    assert isinstance(loaded_data, Data)

def test_serialize_deserialize_data_integrity(tmp_path):
    """Test integrity of data after round trip."""
    file_path = os.path.join(tmp_path, "data.pt")
    data = Data(x=torch.tensor([[1.0], [2.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    torch.save(data, file_path)
    loaded_data = torch.load(file_path)
    assert torch.equal(data.x, loaded_data.x)
    assert torch.equal(data.edge_index, loaded_data.edge_index)

def test_serialize_batch_basic(tmp_path):
    """Test basic serialization of PyG Batch."""
    file_path = os.path.join(tmp_path, "batch.pt")
    data_list = [Data(x=torch.tensor([[1.0]])), Data(x=torch.tensor([[2.0]]))]
    batch = Batch.from_data_list(data_list)
    torch.save(batch, file_path)
    assert os.path.exists(file_path)

def test_deserialize_batch_basic(tmp_path):
    """Test basic deserialization of PyG Batch."""
    file_path = os.path.join(tmp_path, "batch.pt")
    data_list = [Data(x=torch.tensor([[1.0]])), Data(x=torch.tensor([[2.0]]))]
    batch = Batch.from_data_list(data_list)
    torch.save(batch, file_path)
    loaded_batch = torch.load(file_path)
    assert isinstance(loaded_batch, Batch)

def test_serialize_deserialize_batch_integrity(tmp_path):
    """Test integrity of batch after round trip."""
    file_path = os.path.join(tmp_path, "batch.pt")
    data_list = [Data(x=torch.tensor([[1.0]])), Data(x=torch.tensor([[2.0]]))]
    batch = Batch.from_data_list(data_list)
    torch.save(batch, file_path)
    loaded_batch = torch.load(file_path)
    assert loaded_batch.num_graphs == 2
    assert torch.equal(batch.batch, loaded_batch.batch)

def test_serialize_data_with_metadata(tmp_path):
    """Test serialization with arbitrary metadata."""
    file_path = os.path.join(tmp_path, "data_meta.pt")
    data = Data(x=torch.randn(2, 2))
    data.meta_info = {"version": "1.0", "name": "mock"}
    torch.save(data, file_path)
    assert os.path.exists(file_path)

def test_deserialize_data_with_metadata(tmp_path):
    """Test deserialization with arbitrary metadata."""
    file_path = os.path.join(tmp_path, "data_meta.pt")
    data = Data(x=torch.randn(2, 2))
    data.meta_info = {"version": "1.0", "name": "mock"}
    torch.save(data, file_path)
    loaded_data = torch.load(file_path)
    assert loaded_data.meta_info["version"] == "1.0"

def test_serialize_unsupported_type_in_metadata(tmp_path):
    """Test serialization failure modes or resilience with lambda in metadata."""
    file_path = os.path.join(tmp_path, "data_fail.pt")
    data = Data(x=torch.randn(2, 2))
    # PyTorch save can sometimes handle lambdas if pickled by reference, but we test its presence.
    data.func = lambda x: x
    torch.save(data, file_path)
    assert os.path.exists(file_path)

def test_version_metadata_inclusion(tmp_path):
    """Test version string attribute serialization."""
    file_path = os.path.join(tmp_path, "data_version.pt")
    data = Data(x=torch.randn(2, 2), framework_version="2.1")
    torch.save(data, file_path)
    loaded_data = torch.load(file_path)
    assert hasattr(loaded_data, "framework_version")
    assert loaded_data.framework_version == "2.1"

def test_version_metadata_validation_on_load(tmp_path):
    """Test metadata validation pattern mock."""
    file_path = os.path.join(tmp_path, "data_version.pt")
    data = Data(x=torch.randn(2, 2), framework_version="2.1")
    torch.save(data, file_path)
    loaded = torch.load(file_path)
    is_valid = loaded.framework_version == "2.1"
    assert is_valid

def test_serialize_custom_attributes(tmp_path):
    """Test serialization of custom tensor attributes."""
    file_path = os.path.join(tmp_path, "data_custom.pt")
    data = Data(x=torch.randn(2, 2), custom_feat=torch.ones(2))
    torch.save(data, file_path)
    assert os.path.exists(file_path)

def test_deserialize_custom_attributes(tmp_path):
    """Test deserialization of custom tensor attributes."""
    file_path = os.path.join(tmp_path, "data_custom.pt")
    data = Data(x=torch.randn(2, 2), custom_feat=torch.ones(2))
    torch.save(data, file_path)
    loaded = torch.load(file_path)
    assert "custom_feat" in loaded
    assert torch.equal(loaded.custom_feat, torch.ones(2))
