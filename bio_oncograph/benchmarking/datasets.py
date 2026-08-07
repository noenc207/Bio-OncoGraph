import numpy as np
from typing import List, Dict, Tuple, Any, Iterator, Optional
from torch.utils.data import Dataset

class BaseOncoDataset(Dataset):
    """
    Base dataset class for Bio-OncoGraph benchmarking.
    """
    def __init__(self, data: List[Any], labels: List[Any], centers: Optional[List[Any]] = None):
        self.data = data
        self.labels = labels
        self.centers = centers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.centers is not None:
            return self.data[idx], self.labels[idx], self.centers[idx]
        return self.data[idx], self.labels[idx]

class DataSplitter:
    """
    Handles data splitting for various benchmarking protocols including LOCO and Nested CV.
    """
    @staticmethod
    def leave_one_center_out(centers: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Leave-One-Center-Out (LOCO) cross-validation splitter.
        """
        unique_centers = np.unique(centers)
        for center in unique_centers:
            test_idx = np.where(centers == center)[0]
            train_idx = np.where(centers != center)[0]
            yield train_idx, test_idx

    @staticmethod
    def nested_cv(n_samples: int, outer_folds: int = 5, inner_folds: int = 5, seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Nested Cross-Validation splitter.
        Generates outer and inner folds.
        Yields: train_idx, val_idx, test_idx
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        outer_split_size = n_samples // outer_folds
        for i in range(outer_folds):
            outer_test_start = i * outer_split_size
            outer_test_end = n_samples if i == outer_folds - 1 else (i + 1) * outer_split_size
            
            test_idx = indices[outer_test_start:outer_test_end]
            train_val_idx = np.concatenate([indices[:outer_test_start], indices[outer_test_end:]])
            
            n_train_val = len(train_val_idx)
            inner_split_size = n_train_val // inner_folds
            
            for j in range(inner_folds):
                inner_val_start = j * inner_split_size
                inner_val_end = n_train_val if j == inner_folds - 1 else (j + 1) * inner_split_size
                
                val_idx = train_val_idx[inner_val_start:inner_val_end]
                train_idx = np.concatenate([train_val_idx[:inner_val_start], train_val_idx[inner_val_end:]])
                
                yield train_idx, val_idx, test_idx
