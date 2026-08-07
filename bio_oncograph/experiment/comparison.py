"""Module for comparing experimental results."""
from typing import List, Dict, Any
from .statistics import mann_whitney_test, cliffs_delta

class ModelComparison:
    """Utility for comparing models across different runs."""
    
    def __init__(self):
        """Initializes model comparison utility."""
        self.models: Dict[str, List[float]] = {}
        
    def add_model(self, name: str, metrics: List[float]) -> None:
        """Adds model metrics for comparison.
        
        Args:
            name: Name of the model.
            metrics: List of metric results across runs.
        """
        self.models[name] = metrics
        
    def compare_pairs(self, model1: str, model2: str) -> Dict[str, float]:
        """Compares two models.
        
        Args:
            model1: Name of the first model.
            model2: Name of the second model.
            
        Returns:
            Dictionary containing comparison statistics.
        """
        m1 = self.models[model1]
        m2 = self.models[model2]
        
        stat, p_val = mann_whitney_test(m1, m2)
        effect = cliffs_delta(m1, m2)
        
        return {
            "u_stat": stat,
            "p_value": p_val,
            "cliffs_delta": effect
        }
