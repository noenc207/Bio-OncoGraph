"""Configuration for foundation model pretraining."""

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FoundationConfig:
    """Configuration for self-supervised pretraining."""
    encoder_type: str = 'gcn'
    hidden_dim: int = 256
    num_layers: int = 3
    projection_dim: int = 128
    objective_type: str = 'contrastive'
    augmentation_types: Tuple[str, ...] = ('node_dropout', 'edge_dropout')
    node_dropout_rate: float = 0.1
    edge_dropout_rate: float = 0.1
    feature_mask_rate: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    batch_size: int = 32
    temperature: float = 0.07
    momentum: float = 0.999
    seed: int = 42
    deterministic: bool = True
