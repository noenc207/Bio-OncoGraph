"""Reporting utilities for foundation models."""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainingReport:
    """Frozen dataclass representing a training report."""
    num_epochs: int
    final_loss: float
    loss_history: Tuple[float, ...]
    num_graphs: int
    embedding_dim: int
    encoder_type: str
    objective_type: str
    augmentations: Tuple[str, ...]
    is_deterministic: bool
    metadata: Dict[str, Any]

class ReportGenerator:
    """Generator for training reports."""

    def generate(self, trainer_state: Dict[str, Any]) -> TrainingReport:
        """Generate report from trainer state.

        Args:
            trainer_state: Dictionary containing training state information.

        Returns:
            TrainingReport instance.
        """
        try:
            report = TrainingReport(
                num_epochs=trainer_state.get("num_epochs", 0),
                final_loss=trainer_state.get("final_loss", 0.0),
                loss_history=tuple(trainer_state.get("loss_history", [])),
                num_graphs=trainer_state.get("num_graphs", 0),
                embedding_dim=trainer_state.get("embedding_dim", 0),
                encoder_type=trainer_state.get("encoder_type", "Unknown"),
                objective_type=trainer_state.get("objective_type", "Unknown"),
                augmentations=tuple(trainer_state.get("augmentations", [])),
                is_deterministic=trainer_state.get("is_deterministic", False),
                metadata=trainer_state.get("metadata", {})
            )
            logger.info("Successfully generated training report.")
            return report
        except KeyError as e:
            logger.error("Failed to generate report due to missing key: %s", e)
            raise
