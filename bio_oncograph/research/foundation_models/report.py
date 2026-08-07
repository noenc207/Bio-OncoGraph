"""Reporting module for foundation models."""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoundationReport:
    """Report of the foundation model engine state.

    Attributes:
        loaded_models: List of loaded model names.
        cache_size: Number of items in the cache.
        embedding_dim: Expected embedding dimension.
        vlm_name: Name of the Vision-Language Model.
        determinism: Whether execution is deterministic.
    """
    loaded_models: List[str]
    cache_size: int
    embedding_dim: int
    vlm_name: str
    determinism: bool


class ReportGenerator:
    """Generates reports for foundation model engine state."""

    def __init__(self) -> None:
        """Initializes the report generator."""
        logger.debug("Initializing ReportGenerator.")

    def compile_report(
        self,
        loaded_models: List[str],
        cache_size: int,
        embedding_dim: int,
        vlm_name: str,
        determinism: bool,
    ) -> FoundationReport:
        """Compiles the foundation report.

        Args:
            loaded_models: List of loaded model names.
            cache_size: Size of the cache.
            embedding_dim: Dimensionality of embeddings.
            vlm_name: Name of the VLM.
            determinism: Flag indicating deterministic execution.

        Returns:
            A compiled FoundationReport instance.
        """
        logger.info("Compiling foundation report.")
        return FoundationReport(
            loaded_models=loaded_models,
            cache_size=cache_size,
            embedding_dim=embedding_dim,
            vlm_name=vlm_name,
            determinism=determinism,
        )
