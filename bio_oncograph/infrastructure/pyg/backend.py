"""PyTorch Geometric backend facade for Bio-OncoGraph."""

from typing import Any, Dict, List, Optional
import torch
import torch_geometric

from bio_oncograph.infrastructure.pyg.adapters import GraphAdapter
from bio_oncograph.infrastructure.pyg.datasets import InMemoryBioDataset
from bio_oncograph.infrastructure.pyg.dataloaders import create_dataloader

class PyGBackend:
    """Facade for PyTorch Geometric functionality."""

    def __init__(self, device: Optional[str] = None):
        """Initializes the PyG backend.
        
        Args:
            device: Device string to use for backend operations.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.adapter = GraphAdapter()

    def create_dataset(self, data_list: List[Dict[str, Any]], root: str) -> InMemoryBioDataset:
        """Creates an in-memory dataset from a list of data dictionaries.
        
        Args:
            data_list: List of raw data dictionaries.
            root: Root directory for the dataset.
            
        Returns:
            An instance of InMemoryBioDataset.
        """
        return InMemoryBioDataset(root=root, raw_data_list=data_list)

    def get_dataloader(self, dataset: Any, batch_size: int, shuffle: bool = True) -> Any:
        """Creates a PyG DataLoader.
        
        Args:
            dataset: PyG dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.
            
        Returns:
            A PyG DataLoader.
        """
        return create_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
