"""Configuration for the Bio-OncoGraph Open Science module."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class OpenScienceConfig:
    """Configuration settings for open science framework.
    
    Attributes:
        project_name (str): Name of the project.
        version (str): Version of the framework.
        output_dir (str): Output directory for generated artifacts.
        enable_api_docs (bool): Whether to generate API docs.
        enable_tutorials (bool): Whether to generate tutorials.
        enable_examples (bool): Whether to enable examples.
        enable_reproducibility (bool): Whether to track reproducibility metrics.
        seed (int): Global random seed.
    """
    project_name: str = 'BioOncoGraph'
    version: str = '2.0.0'
    output_dir: str = './docs'
    enable_api_docs: bool = True
    enable_tutorials: bool = True
    enable_examples: bool = True
    enable_reproducibility: bool = True
    seed: int = 42

logger.info("OpenScienceConfig initialized.")
