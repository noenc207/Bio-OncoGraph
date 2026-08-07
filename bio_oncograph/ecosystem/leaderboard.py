"""Leaderboard management for scientific evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderboardEntry:
    """An entry in the benchmark leaderboard."""
    model_name: str
    dataset_name: str
    metric_name: str
    score: float
    metadata: Dict[str, Any]
    timestamp: float


class BenchmarkLeaderboard:
    """Manages the benchmark leaderboard."""

    def __init__(self) -> None:
        """Initialize the benchmark leaderboard."""
        self._entries: List[LeaderboardEntry] = []
        logger.info("Initialized BenchmarkLeaderboard")

    def submit(self, entry: LeaderboardEntry) -> None:
        """Submit a new entry to the leaderboard.
        
        Args:
            entry: The leaderboard entry to submit.
        """
        self._entries.append(entry)
        logger.info("Submitted entry for model %s on %s", entry.model_name, entry.dataset_name)

    def get_ranking(self, dataset_name: str, metric_name: str, top_k: int) -> List[LeaderboardEntry]:
        """Get the top-K ranking for a specific dataset and metric.
        
        Args:
            dataset_name: Name of the dataset.
            metric_name: Name of the metric.
            top_k: Number of top entries to return.
            
        Returns:
            A list of the top-K leaderboard entries.
        """
        relevant = [
            e for e in self._entries
            if e.dataset_name == dataset_name and e.metric_name == metric_name
        ]
        relevant.sort(key=lambda e: e.score, reverse=True)
        return relevant[:top_k]

    def compare(
        self, model_a: str, model_b: str, dataset: str, metric: str
    ) -> Dict[str, Any]:
        """Compare two models on a specific dataset and metric.
        
        Args:
            model_a: First model name.
            model_b: Second model name.
            dataset: Dataset name.
            metric: Metric name.
            
        Returns:
            A dictionary with the comparison results.
        """
        scores = {}
        for entry in self._entries:
            if entry.dataset_name == dataset and entry.metric_name == metric:
                if entry.model_name == model_a and model_a not in scores:
                    scores[model_a] = entry.score
                elif entry.model_name == model_b and model_b not in scores:
                    scores[model_b] = entry.score
                    
        return scores

    def export_table(self, dataset: str, metric: str) -> str:
        """Export the leaderboard for a dataset and metric to a markdown table.
        
        Args:
            dataset: Dataset name.
            metric: Metric name.
            
        Returns:
            A string containing the markdown table.
        """
        ranking = self.get_ranking(dataset, metric, top_k=100)
        
        lines = [f"| Rank | Model | {metric} |", "|---|---|---|"]
        for i, entry in enumerate(ranking, 1):
            lines.append(f"| {i} | {entry.model_name} | {entry.score:.4f} |")
            
        return "\n".join(lines)
