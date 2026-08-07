import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional
import hashlib
import numpy as np
import random

class FairBenchmarkProtocol:
    """
    Protocol verifier to ensure exact seeds, splits, and optimizers are used during benchmarking.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seeds(seed)
        
    def set_seeds(self, seed: int) -> None:
        """
        Sets exact seeds for reproducibility across torch, numpy, and python random.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def verify_optimizer(self, optimizer: optim.Optimizer, expected_config: Dict[str, Any]) -> bool:
        """
        Verifies that the optimizer configuration exactly matches the expected benchmark configuration.
        """
        config = optimizer.defaults
        for key, value in expected_config.items():
            if key in config and config[key] != value:
                return False
        return True

    def verify_split(self, train_idx: np.ndarray, test_idx: np.ndarray, expected_train_hash: str, expected_test_hash: str) -> bool:
        """
        Verifies exact dataset splits via hashing.
        """
        train_hash = hashlib.sha256(train_idx.tobytes()).hexdigest()
        test_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
        return train_hash == expected_train_hash and test_hash == expected_test_hash

    def enforce_exact_protocol(self, model: nn.Module, optimizer: optim.Optimizer, train_idx: np.ndarray, test_idx: np.ndarray, expected_configs: Dict[str, Any]) -> bool:
        """
        Enforces all fair benchmarking protocols for a single run.
        """
        self.set_seeds(self.seed)
        
        opt_valid = self.verify_optimizer(optimizer, expected_configs.get("optimizer", {}))
        
        split_valid = True
        if "train_hash" in expected_configs and "test_hash" in expected_configs:
            split_valid = self.verify_split(train_idx, test_idx, expected_configs["train_hash"], expected_configs["test_hash"])
            
        return opt_valid and split_valid
