"""Launcher module for orchestrating distributed HPO workloads."""
import logging
from typing import Callable, Optional, Dict, Any

import optuna
from bio_oncograph.experiment.runner import ExperimentRunner
from bio_oncograph.experiment.storage import get_storage

logger = logging.getLogger(__name__)

def launch_study(
    study_name: str,
    storage_uri: str,
    objective: Callable[[optuna.Trial], float],
    direction: str = "minimize",
    max_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    max_failed_trials: Optional[int] = None,
    max_gpu_hours: Optional[float] = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """Launches or resumes a study with given configurations and constraints.
    
    Args:
        study_name: Name of the study.
        storage_uri: URI for the storage backend (SQLite/Postgres).
        objective: The objective function to optimize.
        direction: Optimization direction ("minimize" or "maximize").
        max_trials: Maximum trials budget.
        timeout: Maximum time budget in seconds.
        max_failed_trials: Maximum number of failed trials allowed.
        max_gpu_hours: Maximum GPU hours allowed.
        load_if_exists: Whether to resume if study already exists.
        
    Returns:
        The optimized Optuna study.
    """
    logger.info(f"Launching study {study_name} on {storage_uri}")
    
    storage = get_storage(storage_uri)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=load_if_exists,
    )
    
    runner = ExperimentRunner(
        study=study,
        objective=objective,
        max_trials=max_trials,
        timeout=timeout,
        max_failed_trials=max_failed_trials,
        max_gpu_hours=max_gpu_hours
    )
    
    runner.run()
    
    return study
