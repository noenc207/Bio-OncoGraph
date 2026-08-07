"""Module for aggregating results across seeds."""
import numpy as np
from typing import List, Dict

class ResultAggregator:
    """Aggregates experiment results across multiple random seeds."""
    
    def __init__(self):
        """Initializes result aggregator."""
        self.results: Dict[str, List[float]] = {}
        
    def add_result(self, key: str, value: float) -> None:
        """Adds a result for a specific key.
        
        Args:
            key: Identifier for the metric/result.
            value: Result value.
        """
        if key not in self.results:
            self.results[key] = []
        self.results[key].append(value)
        
    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Aggregates all results.
        
        Returns:
            Dictionary mapping keys to their aggregated statistics.
        """
        aggregated = {}
        for key, values in self.results.items():
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
                "count": len(values)
            }
        return aggregated
