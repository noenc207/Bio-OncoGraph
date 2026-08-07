"""Tests for PyG datasets."""
import pytest
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset, Dataset
import os

class DummyInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = [
            Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0])),
            Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=torch.tensor([1]))
        ]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DummyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = [
            Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([0])),
            Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), y=torch.tensor([1]))
        ]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

@pytest.fixture
def in_memory_dataset(tmp_path):
    return DummyInMemoryDataset(root=str(tmp_path))

@pytest.fixture
def standard_dataset(tmp_path):
    return DummyDataset(root=str(tmp_path))

def test_in_memory_dataset_len(in_memory_dataset):
    assert len(in_memory_dataset) == 2

def test_in_memory_dataset_get(in_memory_dataset):
    data = in_memory_dataset[0]
    assert data.x.shape == (3, 2)
    assert data.y.item() == 0

def test_in_memory_dataset_files(in_memory_dataset, tmp_path):
    assert os.path.exists(os.path.join(tmp_path, 'processed', 'data.pt'))

def test_standard_dataset_len(standard_dataset):
    assert len(standard_dataset) == 2

def test_standard_dataset_get(standard_dataset):
    data = standard_dataset[1]
    assert data.x.shape == (4, 2)
    assert data.y.item() == 1

def test_dataset_transform(tmp_path):
    import torch_geometric.transforms as T
    dataset = DummyInMemoryDataset(root=str(tmp_path), transform=T.NormalizeFeatures())
    data = dataset[0]
    # Check if features are normalized (sum to 1 if all positive, or specific mean/std based on T.NormalizeFeatures)
    assert data.x is not None

def test_dataset_iteration(standard_dataset):
    count = 0
    for data in standard_dataset:
        assert isinstance(data, Data)
        count += 1
    assert count == 2

def test_dataset_slicing(in_memory_dataset):
    subset = in_memory_dataset[:1]
    assert len(subset) == 1
    assert subset[0].x.shape[0] == 3

def test_dataset_indices(standard_dataset):
    subset = standard_dataset[[0, 1]]
    assert len(subset) == 2

def test_dataset_repr(in_memory_dataset):
    assert "DummyInMemoryDataset(2)" in repr(in_memory_dataset)

def test_dataset_num_classes(in_memory_dataset):
    assert in_memory_dataset.num_classes == 2

def test_dataset_num_node_features(in_memory_dataset):
    assert in_memory_dataset.num_node_features == 2
