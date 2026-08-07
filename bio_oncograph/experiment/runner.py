"""Runner module for executing Optuna trials with budget constraints."""
import time
import logging
from typing import Callable, Optional, Dict, Any

import optuna

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Runner for executing and monitoring an Optuna study with constraints.
    
    Attributes:
        study (optuna.Study): The Optuna study to run.
        objective (Callable): The objective function for optimization.
        max_trials (int | None): Maximum number of trials.
        timeout (float | None): Maximum execution time in seconds.
        max_failed_trials (int | None): Maximum allowed failed trials before aborting.
        max_gpu_hours (float | None): Maximum GPU hours allowed.
    """

    def __init__(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        max_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        max_failed_trials: Optional[int] = None,
        max_gpu_hours: Optional[float] = None,
    ):
        """Initializes the ExperimentRunner."""
        self.study = study
        self.objective = objective
        self.max_trials = max_trials
        self.timeout = timeout
        self.max_failed_trials = max_failed_trials
        self.max_gpu_hours = max_gpu_hours
        self._start_time = time.time()
        self._failed_count = 0
        self._gpu_time_used = 0.0

    def _constraint_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback to check constraints after each trial.
        
        Args:
            study: The current Optuna study.
            trial: The just-finished trial.
        """
        if trial.state == optuna.trial.TrialState.FAIL:
            self._failed_count += 1
            if self.max_failed_trials is not None and self._failed_count >= self.max_failed_trials:
                logger.error(f"Reached maximum failed trials ({self.max_failed_trials}). Stopping.")
                study.stop()
        
        elapsed_time = time.time() - self._start_time
        if self.timeout is not None and elapsed_time >= self.timeout:
            logger.info(f"Reached timeout ({self.timeout}s). Stopping.")
            study.stop()
            
        # Simplistic GPU hours estimation (assumes 1 GPU used continuously per trial)
        if trial.datetime_start and trial.datetime_complete:
            duration_hours = (trial.datetime_complete - trial.datetime_start).total_seconds() / 3600.0
            self._gpu_time_used += duration_hours
            if self.max_gpu_hours is not None and self._gpu_time_used >= self.max_gpu_hours:
                logger.info(f"Reached maximum GPU hours ({self.max_gpu_hours}). Stopping.")
                study.stop()

    def run(self) -> None:
        """Starts the optimization process."""
        logger.info("Starting experiment runner.")
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.max_trials,
                timeout=self.timeout,
                callbacks=[self._constraint_callback],
                catch=(Exception,)
            )
        except Exception as e:
            logger.error(f"Optimization aborted due to error: {e}")
            raise
        logger.info("Experiment runner completed.")
