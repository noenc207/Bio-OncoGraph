"""Dataset card generation and validation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetCard:
    """Comprehensive dataset card."""
    dataset_name: str
    source: str
    preprocessing_steps: List[str]
    split_info: Dict[str, int]
    statistics: Dict[str, Any]
    limitations: List[str]
    license_info: str
    version: str
    timestamp: float


class DatasetCardGenerator:
    """Generator for dataset cards."""

    def generate(
        self, dataset_name: str, source: str, split_info: Dict[str, int], statistics: Dict[str, Any], **kwargs: Any
    ) -> DatasetCard:
        """Generate a dataset card.
        
        Args:
            dataset_name: Name of the dataset.
            source: Source of the dataset.
            split_info: Information about data splits.
            statistics: Dataset statistics.
            **kwargs: Additional fields for the dataset card.
            
        Returns:
            A populated DatasetCard instance.
        """
        card = DatasetCard(
            dataset_name=dataset_name,
            source=source,
            preprocessing_steps=kwargs.get("preprocessing_steps", []),
            split_info=split_info,
            statistics=statistics,
            limitations=kwargs.get("limitations", []),
            license_info=kwargs.get("license_info", "Research Use Only"),
            version=kwargs.get("version", "1.0.0"),
            timestamp=kwargs.get("timestamp", 0.0)
        )
        logger.info("Generated dataset card for %s", dataset_name)
        return card

    def to_markdown(self, card: DatasetCard) -> str:
        """Convert a dataset card to markdown format.
        
        Args:
            card: The dataset card.
            
        Returns:
            A markdown formatted string.
        """
        lines = [
            f"# Dataset Card: {card.dataset_name}",
            f"**Version**: {card.version}",
            f"**Source**: {card.source}",
            f"**License**: {card.license_info}",
            "",
            "## Splits"
        ]
        
        for split, size in card.split_info.items():
            lines.append(f"- {split}: {size}")
            
        lines.extend([
            "",
            "## Statistics"
        ])
        for k, v in card.statistics.items():
            lines.append(f"- {k}: {v}")
            
        lines.extend([
            "",
            "## Preprocessing",
            "\n".join(f"- {step}" for step in card.preprocessing_steps),
            "",
            "## Limitations",
            "\n".join(f"- {lim}" for lim in card.limitations)
        ])
        
        return "\n".join(lines)

    def validate(self, card: DatasetCard) -> List[str]:
        """Validate the completeness of a dataset card.
        
        Args:
            card: The dataset card to validate.
            
        Returns:
            A list of missing or invalid fields.
        """
        issues = []
        if not card.limitations:
            issues.append("Missing limitations.")
        if not card.preprocessing_steps:
            issues.append("Missing preprocessing steps.")
        if sum(card.split_info.values()) == 0:
            issues.append("Split info is empty or zero.")
        return issues
