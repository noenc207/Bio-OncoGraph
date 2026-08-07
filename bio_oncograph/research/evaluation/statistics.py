import logging
import numpy as np
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

class StatisticalEvaluator:
    """Evaluates statistical significance between model results."""
    
    def compare_models(self, metrics_a: Union[List[float], np.ndarray], metrics_b: Union[List[float], np.ndarray], method: str = 'wilcoxon') -> Dict[str, float]:
        """Compares two sets of metrics using the specified statistical test."""
        logger.info(f"Comparing models using {method}")
        a = np.array(metrics_a)
        b = np.array(metrics_b)
        
        if method == 'wilcoxon':
            return self.wilcoxon_signed_rank(a, b)
        elif method == 't_test':
            return self.paired_t_test(a, b)
        else:
            raise ValueError(f"Unknown statistical test method: {method}")

    def paired_t_test(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """Mock implementation of paired t-test."""
        diff = a - b
        t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff))) if len(diff) > 0 and np.std(diff) > 0 else 0.0
        p_val = np.random.rand()
        return {"stat": float(t_stat), "p_value": float(p_val)}

    def wilcoxon_signed_rank(self, a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """Mock implementation of Wilcoxon signed-rank test."""
        diff = a - b
        w_stat = np.sum(np.abs(diff))
        p_val = np.random.rand()
        return {"stat": float(w_stat), "p_value": float(p_val)}

    def cohens_d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates Cohen's d effect size."""
        n1, n2 = len(a), len(b)
        if n1 == 0 or n2 == 0:
            return 0.0
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if n1 + n2 > 2 else 1.0
        if s == 0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / s)
