"""Configuration module for hyperparameter optimization experiments."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for hyperparameter optimization experiments.

    Attributes:
        study_name: The name of the optimization study.
        storage: The database URL for storing the study. If None, in-memory storage is used.
        n_trials: The number of trials to run.
        timeout: Maximum time in seconds for the optimization.
        metrics: A list of metrics to optimize (e.g., 'val_loss', 'roc_auc').
        directions: A list of directions for each metric ('minimize' or 'maximize').
        sampler_name: The name of the sampler to use ('tpe', 'cmaes', 'nsgaii', 'grid', 'random').
        pruner_name: The name of the pruner to use ('median', 'successive_halving', 'hyperband', 'nop').
        sampler_kwargs: Additional keyword arguments for the sampler.
        pruner_kwargs: Additional keyword arguments for the pruner.
    """

    study_name: str = "bio_oncograph_study"
    storage: Optional[str] = None
    n_trials: int = 100
    timeout: Optional[int] = None
    metrics: List[str] = field(default_factory=lambda: ["val_loss"])
    directions: List[str] = field(default_factory=lambda: ["minimize"])
    sampler_name: str = "tpe"
    pruner_name: str = "median"
    sampler_kwargs: Dict[str, any] = field(default_factory=dict)
    pruner_kwargs: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validates the configuration."""
        if len(self.metrics) != len(self.directions):
            raise ValueError(
                f"Number of metrics ({len(self.metrics)}) must match "
                f"number of directions ({len(self.directions)})."
            )
        for direction in self.directions:
            if direction not in ("minimize", "maximize"):
                raise ValueError(
                    f"Direction must be 'minimize' or 'maximize', got '{direction}'."
                )
