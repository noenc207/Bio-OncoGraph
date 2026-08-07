"""Samplers for hyperparameter optimization."""

from typing import Any, Dict
import optuna
from optuna.samplers import (
    TPESampler,
    CmaEsSampler,
    NSGAIISampler,
    GridSampler,
    RandomSampler,
    BaseSampler,
)


def get_sampler(
    sampler_name: str, n_objectives: int = 1, **kwargs: Any
) -> BaseSampler:
    """Creates an Optuna sampler based on the provided name.

    Args:
        sampler_name: The name of the sampler ('tpe', 'cmaes', 'nsgaii', 'grid', 'random').
        n_objectives: The number of objectives. Determines if multi-objective is needed.
        **kwargs: Additional keyword arguments to pass to the sampler constructor.

    Returns:
        An instantiated Optuna sampler.

    Raises:
        ValueError: If the sampler name is unknown or if a single-objective
            sampler is requested for a multi-objective problem.
    """
    sampler_name = sampler_name.lower()

    if n_objectives > 1:
        if sampler_name not in ("nsgaii", "random", "tpe"):
            raise ValueError(
                f"Sampler '{sampler_name}' does not support multi-objective "
                f"optimization (n_objectives={n_objectives}). Use 'nsgaii', 'tpe', or 'random'."
            )

    if sampler_name == "tpe":
        return TPESampler(**kwargs)
    elif sampler_name == "cmaes":
        if n_objectives > 1:
            raise ValueError("CMA-ES does not support multi-objective optimization.")
        return CmaEsSampler(**kwargs)
    elif sampler_name == "nsgaii":
        return NSGAIISampler(**kwargs)
    elif sampler_name == "grid":
        if "search_space" not in kwargs:
            raise ValueError("GridSampler requires a 'search_space' kwarg.")
        return GridSampler(search_space=kwargs.pop("search_space"), **kwargs)
    elif sampler_name == "random":
        return RandomSampler(**kwargs)
    else:
        raise ValueError(
            f"Unknown sampler name: '{sampler_name}'. Supported samplers are: "
            "'tpe', 'cmaes', 'nsgaii', 'grid', 'random'."
        )
