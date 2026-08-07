import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Union

def paired_t_test(metrics_model_a: np.ndarray, metrics_model_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform an exact paired t-test between two sets of metrics across multiple seeds.
    """
    if len(metrics_model_a) != len(metrics_model_b):
        raise ValueError("Metric arrays must have the same length.")
    
    stat, p_value = stats.ttest_rel(metrics_model_a, metrics_model_b)
    return float(stat), float(p_value)

def wilcoxon_signed_rank_test(metrics_model_a: np.ndarray, metrics_model_b: np.ndarray) -> Tuple[float, float]:
    """
    Perform a Wilcoxon signed-rank test between two sets of metrics.
    """
    if len(metrics_model_a) != len(metrics_model_b):
        raise ValueError("Metric arrays must have the same length.")
        
    diff = metrics_model_a - metrics_model_b
    if np.all(diff == 0):
        return 0.0, 1.0 # If all differences are zero
        
    # use method='exact' if possible in newer scipy versions, or exact=True for older ones
    # handle both for robustness
    try:
        stat, p_value = stats.wilcoxon(metrics_model_a, metrics_model_b, method='exact')
    except TypeError:
        # fallback for older scipy versions
        try:
            stat, p_value = stats.wilcoxon(metrics_model_a, metrics_model_b, exact=True)
        except ValueError:
            # happens if N < 10 in older scipy versions for exact=True without zero_method arg
            stat, p_value = stats.wilcoxon(metrics_model_a, metrics_model_b)
            
    return float(stat), float(p_value)

def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    Returns a tuple of (reject_null_hypotheses_list, corrected_p_values_list)
    """
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    m = len(p_values)
    reject = np.zeros(m, dtype=bool)
    corrected_p_values = np.zeros(m)
    
    for k, p in enumerate(sorted_p_values):
        corrected_p = p * (m - k)
        corrected_p_values[sorted_indices[k]] = corrected_p
        if p <= alpha / (m - k):
            reject[sorted_indices[k]] = True
        else:
            break
            
    adj_p = np.minimum(1.0, corrected_p_values)
    return reject.tolist(), adj_p.tolist()

def compare_models(metrics_dict: Dict[str, np.ndarray], baseline_model: str, alpha: float = 0.05) -> Dict[str, Dict[str, Union[float, bool]]]:
    """
    Compare multiple models against a baseline using Wilcoxon signed-rank test with Holm-Bonferroni correction.
    """
    if baseline_model not in metrics_dict:
        raise ValueError(f"Baseline model '{baseline_model}' not found in metrics dictionary.")
        
    baseline_metrics = metrics_dict[baseline_model]
    models_to_compare = [model for model in metrics_dict.keys() if model != baseline_model]
    
    p_values = []
    stats_list = []
    
    for model in models_to_compare:
        stat, p_val = wilcoxon_signed_rank_test(metrics_dict[model], baseline_metrics)
        stats_list.append(stat)
        p_values.append(p_val)
        
    reject_list, corrected_p_values = holm_bonferroni_correction(p_values, alpha)
    
    results = {}
    for i, model in enumerate(models_to_compare):
        results[model] = {
            'statistic': stats_list[i],
            'p_value': p_values[i],
            'corrected_p_value': corrected_p_values[i],
            'significant': reject_list[i]
        }
        
    return results
