"""Runner for benchmarking."""

import time
from typing import Any, Callable, Dict
import numpy as np
import sys

from bio_oncograph.benchmark.benchmark_config import BenchmarkConfig
from bio_oncograph.benchmark.benchmark_report import BenchmarkReport
from bio_oncograph.benchmark.reproducibility import ReproducibilityValidator


class BenchmarkRunner:
    """Runner for evaluating pipeline performance."""

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize the benchmark runner.
        
        Args:
            config: Configuration for the benchmark.
        """
        self.config = config

    def _get_memory_usage(self, obj: Any) -> float:
        """Mock memory tracker returning size in MB.
        
        Args:
            obj: The object to track.
            
        Returns:
            Estimated memory usage in MB.
        """
        return sys.getsizeof(obj) / (1024 * 1024)

    def run_benchmark(self, pipeline_callable: Callable[[Any], Any], inputs: Any) -> BenchmarkReport:
        """Run the benchmark on a given pipeline.
        
        Args:
            pipeline_callable: The pipeline function to benchmark.
            inputs: Inputs to provide to the pipeline.
            
        Returns:
            A BenchmarkReport with the results.
        """
        for _ in range(self.config.warmup_runs):
            _ = pipeline_callable(inputs)

        latencies = []
        memory_usages = []

        for _ in range(self.config.num_runs):
            start_time = time.perf_counter()
            output = pipeline_callable(inputs)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)
            
            if self.config.track_memory:
                memory_usages.append(self._get_memory_usage(output))
                
        latency_arr = np.array(latencies)
        latency_stats = {
            "mean": float(np.mean(latency_arr)),
            "std": float(np.std(latency_arr)),
            "p50": float(np.percentile(latency_arr, 50)),
            "p95": float(np.percentile(latency_arr, 95)),
            "p99": float(np.percentile(latency_arr, 99)),
        }
        
        memory_stats: Dict[str, float] = {}
        if self.config.track_memory and memory_usages:
            mem_arr = np.array(memory_usages)
            memory_stats = {
                "mean": float(np.mean(mem_arr)),
                "max": float(np.max(mem_arr)),
            }
            
        total_time_s = np.sum(latency_arr) / 1000.0
        throughput = self.config.num_runs / total_time_s if total_time_s > 0 else 0.0
        
        is_deterministic = True
        if self.config.validate_reproducibility:
            validator = ReproducibilityValidator()
            is_deterministic = validator.check_determinism(
                pipeline_callable, inputs, num_trials=self.config.num_runs
            )
            
        return BenchmarkReport(
            latency_ms=latency_stats,
            memory_mb=memory_stats,
            throughput=throughput,
            is_deterministic=is_deterministic,
        )
