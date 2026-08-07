"""Aggregation module for scientific evaluation results."""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class ExperimentRun:
    """Represents a single experiment run."""
    def __init__(self, run_id: str, metrics: Dict[str, float]):
        """Initialize an experiment run.

        Args:
            run_id: Identifier for the run.
            metrics: Dictionary of metric names to values.
        """
        self.run_id = run_id
        self.metrics = metrics


class ResultAggregator:
    """Aggregator for experimental results."""

    @staticmethod
    def aggregate_metrics(runs: List[ExperimentRun]) -> Dict[str, Dict[str, float]]:
        """Compute mean, standard deviation, and confidence intervals for metrics.

        Args:
            runs: A list of ExperimentRun instances.

        Returns:
            A dictionary mapping metric names to their aggregated statistics.
        """
        logger.info("Aggregating metrics for %d runs", len(runs))
        if not runs:
            return {}

        metric_keys = runs[0].metrics.keys()
        aggregated: Dict[str, Dict[str, float]] = {}

        for key in metric_keys:
            values = [run.metrics[key] for run in runs if key in run.metrics]
            if values:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                ci_val = 1.96 * std_val / np.sqrt(len(values))
                aggregated[key] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_95": ci_val
                }

        return aggregated

    @staticmethod
    def cross_dataset_aggregation(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate results across multiple datasets.

        Args:
            results: Dictionary mapping dataset names to their metrics.

        Returns:
            A dictionary containing cross-dataset aggregated metrics.
        """
        logger.info("Performing cross-dataset aggregation")
        aggregated: Dict[str, float] = {}
        if not results:
            return aggregated

        datasets = list(results.keys())
        first_dataset_metrics = results[datasets[0]].keys()

        for metric in first_dataset_metrics:
            values = [results[ds].get(metric, 0.0) for ds in datasets]
            aggregated[metric] = float(np.mean(values))

        return aggregated
