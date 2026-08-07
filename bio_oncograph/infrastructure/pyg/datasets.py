"""Custom PyTorch Geometric datasets for Bio-OncoGraph."""

import os
from typing import Any, Callable, Dict, List, Optional
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data

from bio_oncograph.infrastructure.pyg.adapters import GraphAdapter

class InMemoryBioDataset(InMemoryDataset):
    """In-memory PyG dataset for Bio-OncoGraph data."""

    def __init__(
        self,
        root: str,
        raw_data_list: Optional[List[Dict[str, Any]]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ) -> None:
        """Initializes the dataset.
        
        Args:
            root: Root directory where the dataset should be saved.
            raw_data_list: List of raw data dictionaries to process.
            transform: A function/transform that takes in a Data object and returns a transformed version.
            pre_transform: A function/transform that takes in a Data object and returns a transformed version before saving.
            pre_filter: A function that takes in a Data object and returns a boolean value.
        """
        self.raw_data_list = raw_data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files in the dataset."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files in the dataset."""
        return ['data.pt']

    def download(self) -> None:
        """Downloads the dataset. Not implemented for this custom dataset."""
        pass

    def process(self) -> None:
        """Processes raw data into PyG Data objects and saves them."""
        if self.raw_data_list is None:
            raise ValueError("raw_data_list must be provided to process data.")
            
        adapter = GraphAdapter()
        data_list = [adapter.to_pyg(data_dict) for data_dict in self.raw_data_list]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
