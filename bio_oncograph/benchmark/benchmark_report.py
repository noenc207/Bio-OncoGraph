"""Report for benchmarking."""

import dataclasses
from typing import Dict


@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """Report generated from running a benchmark.
    
    Attributes:
        latency_ms: Dictionary containing latency statistics (e.g., mean, std, p50, p99).
        memory_mb: Dictionary containing memory usage statistics.
        throughput: Throughput in items/operations per second.
        is_deterministic: Whether the pipeline output was deterministic.
    """
    latency_ms: Dict[str, float]
    memory_mb: Dict[str, float]
    throughput: float
    is_deterministic: bool
