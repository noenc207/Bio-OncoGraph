"""Hydra entrypoint for launching HPO workloads."""
import logging
import os
import sys
import optuna

# Ensure module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bio_oncograph.experiment.launcher import launch_study
from bio_oncograph.experiment.registry import Registry
from bio_oncograph.experiment.utils import capture_reproducibility_snapshot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dummy_objective(trial: optuna.Trial) -> float:
    """Dummy objective for demonstration."""
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

def main() -> None:
    """Main execution entrypoint for hyperparameter tuning."""
    # In a real setup, Hydra would populate this config
    config = {
        "study_name": "bio_oncograph_study",
        "storage_uri": "sqlite:///tuning.db",
        "direction": "minimize",
        "max_trials": 100,
        "timeout": 3600.0,
        "max_failed_trials": 5,
        "max_gpu_hours": 12.0,
        "output_dir": "./outputs/tune"
    }
    
    capture_reproducibility_snapshot(config["output_dir"])
    
    registry = Registry(os.path.join(config["output_dir"], "registry"))
    registry.register_experiment(config["study_name"], config)
    
    def tracked_objective(trial: optuna.Trial) -> float:
        registry.register_trial(config["study_name"], trial.number, trial.params)
        return dummy_objective(trial)
    
    launch_study(
        study_name=config["study_name"],
        storage_uri=config["storage_uri"],
        objective=tracked_objective,
        direction=config["direction"],
        max_trials=config["max_trials"],
        timeout=config["timeout"],
        max_failed_trials=config["max_failed_trials"],
        max_gpu_hours=config["max_gpu_hours"]
    )
    logger.info("Tuning complete.")

if __name__ == "__main__":
    main()
