import dataclasses
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class ExperimentRun:
    """Represents a single experiment run."""
    run_id: str
    timestamp: str
    model_name: str
    dataset_name: str
    metrics: Dict[str, float]
    seed: int

class ExperimentRegistry:
    """Registry to save and load experiment runs."""
    
    def __init__(self, registry_path: str = "experiments.json") -> None:
        """Initialize the registry."""
        self.registry_path = registry_path
        self.runs: List[ExperimentRun] = []

    def save_experiment(self, run: ExperimentRun) -> None:
        """Saves an experiment run to the registry."""
        self.runs.append(run)
        logger.info(f"Saved experiment {run.run_id} to registry.")

    def load_experiments(self) -> List[ExperimentRun]:
        """Loads experiment runs from the registry."""
        logger.info("Loading experiments from registry.")
        return self.runs
