"""Statistical utility functions for experiments."""
import numpy as np
import scipy.stats as stats
from typing import Tuple, List, Union

def paired_ttest(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """Computes paired t-test.
    
    Args:
        a: First array of values.
        b: Second array of values.
        
    Returns:
        Tuple of t-statistic and p-value.
    """
    stat, p_val = stats.ttest_rel(a, b)
    return float(stat), float(p_val)

def wilcoxon_test(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """Computes Wilcoxon signed-rank test.
    
    Args:
        a: First array of values.
        b: Second array of values.
        
    Returns:
        Tuple of statistic and p-value.
    """
    stat, p_val = stats.wilcoxon(a, b)
    return float(stat), float(p_val)

def mann_whitney_test(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """Computes Mann-Whitney U test.
    
    Args:
        a: First array of values.
        b: Second array of values.
        
    Returns:
        Tuple of statistic and p-value.
    """
    stat, p_val = stats.mannwhitneyu(a, b)
    return float(stat), float(p_val)

def bootstrap_ci(data: Union[List[float], np.ndarray], num_samples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes bootstrap confidence interval for the mean.
    
    Args:
        data: Array of values.
        num_samples: Number of bootstrap samples.
        ci: Confidence interval level.
        
    Returns:
        Tuple of lower and upper confidence interval bounds.
    """
    data = np.array(data)
    resamples = np.random.choice(data, size=(num_samples, len(data)), replace=True)
    means = np.mean(resamples, axis=1)
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lower, upper

def cohens_d(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> float:
    """Computes Cohen's d effect size.
    
    Args:
        a: First array of values.
        b: Second array of values.
        
    Returns:
        Cohen's d value.
    """
    a = np.array(a)
    b = np.array(b)
    n1, n2 = len(a), len(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / s)

def cliffs_delta(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> float:
    """Computes Cliff's Delta effect size.
    
    Args:
        a: First array of values.
        b: Second array of values.
        
    Returns:
        Cliff's Delta value.
    """
    a = np.array(a)
    b = np.array(b)
    m, n = len(a), len(b)
    
    delta = 0
    for i in a:
        for j in b:
            if i > j:
                delta += 1
            elif i < j:
                delta -= 1
    return float(delta / (m * n))

def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Applies Holm-Bonferroni correction to p-values.
    
    Args:
        p_values: List of p-values to correct.
        alpha: Significance level.
        
    Returns:
        Tuple of boolean list (reject null hypothesis) and adjusted p-values.
    """
    sorted_indices = np.argsort(p_values)
    m = len(p_values)
    adjusted_p_values = np.zeros(m)
    reject = np.zeros(m, dtype=bool)
    
    for k, idx in enumerate(sorted_indices):
        adj_p = p_values[idx] * (m - k)
        adjusted_p_values[idx] = min(adj_p, 1.0)
        
    for k, idx in enumerate(sorted_indices):
        if k > 0:
            adjusted_p_values[idx] = max(adjusted_p_values[idx], adjusted_p_values[sorted_indices[k-1]])
        reject[idx] = adjusted_p_values[idx] <= alpha
        
    return reject.tolist(), adjusted_p_values.tolist()
