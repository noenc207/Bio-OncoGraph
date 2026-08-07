"""Benchmarking repository and tracking utilities."""

import logging
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BenchmarkEntry:
    """A single benchmark registration entry."""
    name: str
    dataset: str
    metric: str
    baseline_score: float

@dataclass(frozen=True)
class BenchmarkComparison:
    """A comparison against a benchmark baseline."""
    benchmark_name: str
    baseline: float
    new_score: float
    improvement: float
    is_better: bool

class BenchmarkRepository:
    """Repository to track and compare benchmark results."""
    
    def __init__(self) -> None:
        """Initializes the benchmark repository."""
        self._benchmarks: Dict[str, BenchmarkEntry] = {}
        logger.info("BenchmarkRepository initialized.")
        
    def register_benchmark(self, name: str, dataset: str, metric: str, baseline_score: float) -> None:
        """Registers a new benchmark.
        
        Args:
            name (str): Unique benchmark name.
            dataset (str): Name of the dataset used.
            metric (str): The metric used for scoring.
            baseline_score (float): The baseline score to beat.
        """
        logger.info("Registering benchmark: %s", name)
        self._benchmarks[name] = BenchmarkEntry(
            name=name,
            dataset=dataset,
            metric=metric,
            baseline_score=baseline_score
        )
        
    def get_benchmark(self, name: str) -> BenchmarkEntry:
        """Retrieves a registered benchmark.
        
        Args:
            name (str): Name of the benchmark.
            
        Returns:
            BenchmarkEntry: The benchmark entry.
            
        Raises:
            KeyError: If the benchmark does not exist.
        """
        if name not in self._benchmarks:
            logger.error("Benchmark not found: %s", name)
            raise KeyError(f"Benchmark '{name}' not found.")
        return self._benchmarks[name]
        
    def list_benchmarks(self) -> List[BenchmarkEntry]:
        """Lists all registered benchmarks.
        
        Returns:
            List[BenchmarkEntry]: List of all benchmark entries.
        """
        return list(self._benchmarks.values())
        
    def compare_results(self, name: str, new_score: float) -> BenchmarkComparison:
        """Compares a new score against the registered benchmark baseline.
        
        Args:
            name (str): Name of the benchmark.
            new_score (float): The new score to evaluate.
            
        Returns:
            BenchmarkComparison: Result of the comparison.
        """
        logger.info("Comparing results for benchmark: %s", name)
        entry = self.get_benchmark(name)
        
        # Assuming higher is better for generic metrics; in practice this might depend on the metric.
        improvement = new_score - entry.baseline_score
        is_better = improvement > 0
        
        return BenchmarkComparison(
            benchmark_name=name,
            baseline=entry.baseline_score,
            new_score=new_score,
            improvement=improvement,
            is_better=is_better
        )
