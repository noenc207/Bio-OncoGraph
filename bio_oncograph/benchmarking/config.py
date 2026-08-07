from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class BaselineConfig:
    """Configuration for a specific baseline model."""
    name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    use_gpu: bool = True
    batch_size: int = 1
    num_workers: int = 4

@dataclass
class BenchmarkConfig:
    """Configuration for the overall benchmark execution."""
    dataset_name: str
    dataset_path: str
    baselines: Dict[str, BaselineConfig] = field(default_factory=dict)
    output_dir: str = "./benchmark_results"
    metrics: list = field(default_factory=lambda: ["accuracy", "auc", "f1"])
