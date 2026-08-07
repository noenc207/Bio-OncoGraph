"""Model card generation and validation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCard:
    """Comprehensive model card."""
    model_name: str
    architecture: str
    datasets: List[str]
    training_details: Dict[str, Any]
    metrics: Dict[str, float]
    limitations: List[str]
    intended_use: str
    ethical_considerations: str
    version: str
    timestamp: float


class ModelCardGenerator:
    """Generator for model cards."""

    def generate(
        self, model_name: str, architecture: str, datasets: List[str], metrics: Dict[str, float], **kwargs: Any
    ) -> ModelCard:
        """Generate a model card.
        
        Args:
            model_name: Name of the model.
            architecture: Model architecture description.
            datasets: List of datasets used.
            metrics: Dictionary of evaluation metrics.
            **kwargs: Additional fields for the model card.
            
        Returns:
            A populated ModelCard instance.
        """
        card = ModelCard(
            model_name=model_name,
            architecture=architecture,
            datasets=datasets,
            training_details=kwargs.get("training_details", {}),
            metrics=metrics,
            limitations=kwargs.get("limitations", []),
            intended_use=kwargs.get("intended_use", "Scientific research only."),
            ethical_considerations=kwargs.get("ethical_considerations", "No clinical claims."),
            version=kwargs.get("version", "1.0.0"),
            timestamp=kwargs.get("timestamp", 0.0)
        )
        logger.info("Generated model card for %s", model_name)
        return card

    def to_markdown(self, card: ModelCard) -> str:
        """Convert a model card to markdown format.
        
        Args:
            card: The model card.
            
        Returns:
            A markdown formatted string.
        """
        lines = [
            f"# Model Card: {card.model_name}",
            f"**Version**: {card.version}",
            f"**Architecture**: {card.architecture}",
            "",
            "## Intended Use",
            card.intended_use,
            "",
            "## Datasets",
            ", ".join(card.datasets),
            "",
            "## Metrics"
        ]
        
        for k, v in card.metrics.items():
            lines.append(f"- {k}: {v:.4f}")
            
        lines.extend([
            "",
            "## Limitations",
            "\n".join(f"- {lim}" for lim in card.limitations),
            "",
            "## Ethical Considerations",
            card.ethical_considerations
        ])
        
        return "\n".join(lines)

    def validate(self, card: ModelCard) -> List[str]:
        """Validate the completeness of a model card.
        
        Args:
            card: The model card to validate.
            
        Returns:
            A list of missing or invalid fields.
        """
        issues = []
        if not card.limitations:
            issues.append("Missing limitations.")
        if not card.datasets:
            issues.append("Missing datasets.")
        if not card.metrics:
            issues.append("Missing metrics.")
        return issues
