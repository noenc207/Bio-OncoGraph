"""PyTorch initialization and reproducibility module."""
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The integer seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
