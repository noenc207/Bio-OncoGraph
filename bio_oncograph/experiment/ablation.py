"""Module for comparing ablation configurations."""
from typing import List, Dict, Any

class AblationStudy:
    """Represents an ablation study for model components."""
    
    def __init__(self, baseline_results: List[float], baseline_name: str = "baseline"):
        """Initializes ablation study.
        
        Args:
            baseline_results: List of metric results for the baseline model.
            baseline_name: Name of the baseline model.
        """
        self.baseline_results = baseline_results
        self.baseline_name = baseline_name
        self.ablation_results: Dict[str, List[float]] = {}
        
    def add_ablation(self, name: str, results: List[float]) -> None:
        """Adds ablation results.
        
        Args:
            name: Name of the ablated configuration.
            results: List of metric results.
        """
        self.ablation_results[name] = results
        
    def compare(self) -> Dict[str, Any]:
        """Compares ablations against baseline.
        
        Returns:
            Dictionary containing comparison statistics.
        """
        from .statistics import paired_ttest, cohens_d
        
        comparisons = {}
        for name, results in self.ablation_results.items():
            stat, p_val = paired_ttest(self.baseline_results, results)
            effect_size = cohens_d(self.baseline_results, results)
            comparisons[name] = {
                "t_stat": stat,
                "p_value": p_val,
                "cohens_d": effect_size
            }
        return comparisons
