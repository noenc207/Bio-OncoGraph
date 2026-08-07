"""Configuration for foundation models."""

import dataclasses

@dataclasses.dataclass(frozen=True)
class FoundationModelConfig:
    """Configuration for foundation models."""
    image_encoder_name: str = 'dinov2'
    text_encoder_name: str = 'clip'
    graph_encoder_name: str = 'graph-mae'
    vlm_name: str = 'llava-med'
    embedding_dim: int = 768
    cache_dir: str = './cache'
    deterministic: bool = True
    seed: int = 42
