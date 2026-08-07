"""
Reproducibility utilities for Bio-OncoGraph.

This module provides ``seed_everything`` which locks ALL sources of
randomness (Python, NumPy, PyTorch CPU, PyTorch CUDA, cuDNN) to
guarantee bitwise-deterministic training when combined with
``torch.use_deterministic_algorithms(True)``.

Design reference: Master Design § Versioning & Reproducibility.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, workers: bool = True) -> int:
    """
    Seed every random number generator used by the framework.

    This function ensures full reproducibility by locking:
      - Python's ``random`` module.
      - NumPy's global RNG.
      - PyTorch's CPU and CUDA RNGs.
      - cuDNN's benchmark and deterministic flags.
      - (Optionally) the ``PL_GLOBAL_SEED`` env-var consumed by
        DataLoader workers.

    Args:
        seed: The integer seed value.  Must be non-negative.
        workers: If ``True``, set the ``PL_GLOBAL_SEED`` environment
                 variable so that DataLoader workers (``num_workers > 0``)
                 inherit deterministic behaviour.

    Returns:
        The seed that was applied (echoed for logging convenience).

    Raises:
        ValueError: If *seed* is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic cuDNN behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if workers:
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    return seed
