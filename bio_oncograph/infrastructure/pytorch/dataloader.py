"""Data loading utilities for Bio-OncoGraph PyTorch backend."""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class BioOncoGraphDataset(Dataset):
    """Base dataset class for Bio-OncoGraph.
    
    Attributes:
        data (List[Any]): List of data items.
        transform (Optional[Callable]): Optional transform to be applied on a sample.
    """
    
    def __init__(self, data: List[Any], transform: Optional[Callable] = None) -> None:
        """Initializes the dataset.
        
        Args:
            data: A list containing the data items.
            transform: An optional callable to transform the data items.
        """
        self.data = data
        self.transform = transform
        
    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Any:
        """Generates one sample of data.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            The sample at the specified index, optionally transformed.
        """
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def seed_worker(worker_id: int) -> None:
    """Sets the seed for dataloader workers to ensure reproducibility.
    
    Args:
        worker_id: The ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def custom_collate_fn(batch: List[Any]) -> Any:
    """Custom collation function for Bio-OncoGraph data.
    
    Falls back to default_collate but allows for specific batching logic
    if the data structures require custom handling (e.g., graphs, variable length sequences).
    
    Args:
        batch: A list of samples to collate into a batch.
        
    Returns:
        The collated batch.
    """
    # Implement custom collation logic here if needed.
    return default_collate(batch)

def create_dataloader(
    dataset: Dataset,
    config: Dict[str, Any]
) -> DataLoader:
    """Creates a DataLoader with standard Bio-OncoGraph configurations.
    
    Args:
        dataset: The PyTorch dataset to load data from.
        config: Configuration dictionary for the DataLoader.
            Supports keys like 'batch_size', 'shuffle', 'num_workers', 'pin_memory'.
            
    Returns:
        An instantiated PyTorch DataLoader.
    """
    generator = torch.Generator()
    seed = config.get("seed")
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator = None
        
    return DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=config.get("shuffle", True),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        collate_fn=config.get("collate_fn", custom_collate_fn),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=config.get("drop_last", False)
    )
