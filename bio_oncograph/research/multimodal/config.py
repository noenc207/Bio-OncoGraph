"""Configuration classes for multi-modal learning research."""

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class MultiModalConfig:
    """Configuration for multi-modal feature extraction, alignment, and fusion.

    Attributes:
        supported_modalities: A tuple of strings specifying the supported modalities.
        fusion_strategy: The strategy to use for fusing multiple modalities.
        alignment_strategy: The strategy for aligning features from different modalities.
        missing_strategy: The strategy to handle missing modalities.
        embedding_dim: The standard embedding dimension across all aligned modalities.
        seed: Random seed for reproducibility.
    """
    supported_modalities: Tuple[str, ...] = (
        "histopathology",
        "clinical",
        "genomics",
        "radiology",
        "proteomics",
    )
    fusion_strategy: str = "early"
    alignment_strategy: str = "latent"
    missing_strategy: str = "mask"
    embedding_dim: int = 256
    seed: int = 42
