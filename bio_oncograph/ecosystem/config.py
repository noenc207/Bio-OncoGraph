"""Configuration module for the Bio-OncoGraph ecosystem."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EcosystemConfig:
    """Configuration for the scientific ecosystem."""
    project_name: str = "BioOncoGraph"
    version: str = "2.0.0"
    output_dir: str = "./publications"
    figure_format: str = "png"
    figure_dpi: int = 300
    table_format: str = "latex"
    seed: int = 42
    deterministic: bool = True
