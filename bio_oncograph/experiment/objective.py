"""Objective function definition for hyperparameter optimization."""

from typing import Any, Callable, Dict, List, Tuple, Union
import optuna

from .search_space import SearchSpace

# Type alias for a function that runs an engine and returns metrics.
# In a real scenario, this would be a callable that takes a dictionary of
# hyperparameters and returns a dictionary of validation metrics.
RuntimeEngineType = Callable[[Dict[str, Any]], Dict[str, float]]


def create_objective(
    engine_runner: RuntimeEngineType,
    search_space: SearchSpace,
    metrics: List[str],
) -> Callable[[optuna.Trial], Union[float, Tuple[float, ...]]]:
    """Creates an objective function for Optuna.

    The returned function maps suggested hyperparameters from the search space
    to execution of the RuntimeEngine, extracting the required evaluation metrics.

    Args:
        engine_runner: A callable that accepts a dictionary of hyperparameters
            and runs the training/evaluation engine, returning a dictionary of
            computed metrics (e.g., {'val_loss': 0.5, 'roc_auc': 0.85, ...}).
        search_space: The defined SearchSpace from which to sample hyperparameters.
        metrics: A list of metric names to extract and optimize. If multiple metrics
            are provided, the objective function returns a tuple (for multi-objective
            optimization, such as with NSGA-II).

    Returns:
        A callable objective function that takes an optuna.Trial and returns
        either a single float (for single-objective) or a tuple of floats
        (for multi-objective).
    """

    def objective(trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
        """The core objective function evaluated by Optuna.

        Args:
            trial: The current optimization trial.

        Returns:
            The metric(s) to optimize.
        """
        # Suggest hyperparameters based on the defined search space
        hyperparameters = search_space.suggest(trial)

        # Execute the runtime engine with the sampled hyperparameters
        # It is expected that the engine runner might handle trial pruning
        # internally via callbacks.
        results = engine_runner(hyperparameters)

        # Extract the requested metrics
        objective_values = []
        for metric in metrics:
            if metric not in results:
                raise ValueError(
                    f"Metric '{metric}' not found in engine results. "
                    f"Available metrics: {list(results.keys())}"
                )
            objective_values.append(results[results.keys().index(metric)] if isinstance(results, list) else results[metric])
            
        if len(objective_values) == 1:
            return objective_values[0]
        return tuple(objective_values)

    return objective
