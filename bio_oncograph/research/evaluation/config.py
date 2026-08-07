import dataclasses
from typing import Tuple

@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for scientific evaluation."""
    benchmark_datasets: Tuple[str, ...]
    num_seeds: int = 5
    cv_folds: int = 5
    alpha: float = 0.05
    stat_test: str = 'wilcoxon'
    ablation_types: Tuple[str, ...] = ('module', 'feature', 'modality')
    artifact_dir: str = './artifacts'
    seed: int = 42
