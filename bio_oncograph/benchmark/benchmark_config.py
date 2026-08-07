"""Configuration for benchmarking."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for running benchmarks.
    
    Attributes:
        num_runs: The number of times to run the benchmark.
        warmup_runs: The number of warmup runs before timing.
        track_memory: Whether to track memory usage.
        validate_reproducibility: Whether to validate determinism.
    """
    num_runs: int = 10
    warmup_runs: int = 2
    track_memory: bool = False
    validate_reproducibility: bool = True
