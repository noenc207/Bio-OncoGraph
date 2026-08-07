"""Experiment tracking and management for the scientific ecosystem."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a specific experiment."""
    name: str
    model_name: str
    dataset_name: str
    hyperparams: Dict[str, Any]
    seed: int
    tags: List[str]
    timestamp: float


@dataclass(frozen=True)
class ExperimentResult:
    """Results from an experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    artifacts: Dict[str, str]
    duration_seconds: float
    timestamp: float


class ExperimentManager:
    """Manages experiment tracking and retrieval."""

    def __init__(self) -> None:
        """Initialize the ExperimentManager."""
        self._experiments: Dict[str, ExperimentResult] = {}
        logger.info("Initialized ExperimentManager")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment tracking entry.
        
        Args:
            config: The experiment configuration.
            
        Returns:
            A unique experiment ID.
        """
        experiment_id = str(uuid.uuid4())
        logger.info("Created experiment with ID: %s", experiment_id)
        return experiment_id

    def log_result(self, experiment_id: str, result: ExperimentResult) -> None:
        """Log the result of an experiment.
        
        Args:
            experiment_id: The unique experiment ID.
            result: The experiment results.
        """
        self._experiments[experiment_id] = result
        logger.info("Logged result for experiment ID: %s", experiment_id)

    def get_experiment(self, experiment_id: str) -> ExperimentResult:
        """Retrieve an experiment result by ID.
        
        Args:
            experiment_id: The unique experiment ID.
            
        Returns:
            The experiment result.
            
        Raises:
            KeyError: If the experiment ID does not exist.
        """
        if experiment_id not in self._experiments:
            logger.error("Experiment ID %s not found.", experiment_id)
            raise KeyError(f"Experiment ID {experiment_id} not found.")
        return self._experiments[experiment_id]

    def list_experiments(self, tags: Optional[List[str]] = None) -> List[str]:
        """List experiment IDs, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by.
            
        Returns:
            A list of experiment IDs.
        """
        if not tags:
            return list(self._experiments.keys())
        
        tag_set = set(tags)
        return [
            exp_id for exp_id, res in self._experiments.items()
            if tag_set.issubset(set(res.config.tags))
        ]

    def compare_experiments(self, ids: List[str], metric: str) -> Dict[str, float]:
        """Compare multiple experiments based on a specific metric.
        
        Args:
            ids: List of experiment IDs to compare.
            metric: The metric to compare on.
            
        Returns:
            A dictionary mapping experiment ID to metric value.
        """
        comparison = {}
        for exp_id in ids:
            try:
                result = self.get_experiment(exp_id)
                comparison[exp_id] = result.metrics.get(metric, float("nan"))
            except KeyError:
                logger.warning("Skipping missing experiment ID: %s", exp_id)
        return comparison
