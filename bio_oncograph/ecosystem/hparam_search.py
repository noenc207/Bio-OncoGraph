"""Hyperparameter search strategies and execution engine."""

import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from bio_oncograph.ecosystem.experiments import ExperimentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSpace:
    """Defines a search space for a single hyperparameter."""
    param_name: str
    values: List[Any]


class GridSearchStrategy:
    """Grid search strategy for hyperparameter optimization."""

    def generate_configs(self, spaces: List[SearchSpace]) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters in the search space.
        
        Args:
            spaces: List of search spaces.
            
        Returns:
            A list of configuration dictionaries.
        """
        param_names = [s.param_name for s in spaces]
        value_lists = [s.values for s in spaces]
        
        configs = []
        for combination in itertools.product(*value_lists):
            config = dict(zip(param_names, combination))
            configs.append(config)
            
        logger.info("Generated %d configurations using grid search.", len(configs))
        return configs


class RandomSearchStrategy:
    """Random search strategy for hyperparameter optimization."""

    def generate_configs(
        self, spaces: List[SearchSpace], n_trials: int, seed: int
    ) -> List[Dict[str, Any]]:
        """Generate random configurations from the search space.
        
        Args:
            spaces: List of search spaces.
            n_trials: Number of trials to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            A list of configuration dictionaries.
        """
        rng = random.Random(seed)
        configs = []
        
        for _ in range(n_trials):
            config = {s.param_name: rng.choice(s.values) for s in spaces}
            configs.append(config)
            
        logger.info("Generated %d configurations using random search.", len(configs))
        return configs


class HparamSearchEngine:
    """Engine for executing hyperparameter searches."""

    def run_search(
        self, strategy: Any, experiment_fn: Callable[[Dict[str, Any]], ExperimentResult]
    ) -> List[ExperimentResult]:
        """Run the hyperparameter search using the provided strategy.
        
        Args:
            strategy: The search strategy iterable (configs).
            experiment_fn: Function that takes a config and returns an ExperimentResult.
            
        Returns:
            A list of experiment results.
        """
        logger.info("Starting hyperparameter search.")
        results = []
        for config in strategy:
            logger.info("Running experiment with config: %s", config)
            result = experiment_fn(config)
            results.append(result)
        return results

    def best_config(
        self, results: List[ExperimentResult], metric: str, maximize: bool
    ) -> ExperimentResult:
        """Find the best configuration based on a specific metric.
        
        Args:
            results: List of experiment results.
            metric: The metric to evaluate.
            maximize: Whether to maximize or minimize the metric.
            
        Returns:
            The best experiment result.
            
        Raises:
            ValueError: If the results list is empty.
        """
        if not results:
            raise ValueError("No results provided to find the best configuration.")
            
        def key_fn(res: ExperimentResult) -> float:
            return res.metrics.get(metric, float("-inf") if maximize else float("inf"))
            
        best = max(results, key=key_fn) if maximize else min(results, key=key_fn)
        logger.info("Found best config for metric %s", metric)
        return best
