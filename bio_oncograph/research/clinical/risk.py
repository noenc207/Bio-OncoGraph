import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RiskOutput:
    """Output of a risk stratification."""
    risk_scores: np.ndarray
    risk_groups: np.ndarray
    thresholds: Tuple[float, ...]
    metadata: Dict[str, str] = field(default_factory=dict)

class RiskStratifier:
    """Stratifies continuous risk scores into categorical risk groups."""

    def compute_thresholds(self, risk_scores: np.ndarray, num_groups: int) -> Tuple[float, ...]:
        """Computes quantile-based thresholds for risk stratification."""
        if num_groups < 2:
            raise ValueError("Number of groups must be at least 2.")
        quantiles = np.linspace(0, 1, num_groups + 1)[1:-1]
        thresholds = np.quantile(risk_scores, quantiles)
        return tuple(float(t) for t in thresholds)

    def stratify(self, risk_scores: np.ndarray, num_groups: int, thresholds: Optional[Tuple[float, ...]] = None) -> np.ndarray:
        """Assigns risk groups based on computed or provided thresholds."""
        if thresholds is None:
            thresholds = self.compute_thresholds(risk_scores, num_groups)
        else:
            if len(thresholds) != num_groups - 1:
                raise ValueError("Number of thresholds must be num_groups - 1.")

        groups = np.zeros_like(risk_scores, dtype=int)
        for i, threshold in enumerate(thresholds):
            groups[risk_scores > threshold] = i + 1
        return groups

    def group_statistics(self, risk_scores: np.ndarray, groups: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Computes mean, std, and count per risk group."""
        stats = {}
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = (groups == g)
            group_scores = risk_scores[mask]
            stats[int(g)] = {
                "mean": float(np.mean(group_scores)),
                "std": float(np.std(group_scores)),
                "count": float(np.sum(mask))
            }
        return stats
