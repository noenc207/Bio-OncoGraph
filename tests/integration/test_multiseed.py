import pytest
import math
import numpy as np
from typing import List, Dict
from bio_oncograph.core.evaluator import MultiSeedAggregator

def get_expected_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_95": 0.0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    ci_95 = 1.96 * std / math.sqrt(n) if n > 0 else 0.0
    return {"mean": mean, "std": std, "ci_95": ci_95}

seeds_sets = [
    [42, 123, 777, 2026, 3407],
    [42, 123, 777],
    [2026, 3407],
    [42],
    [123, 777, 2026, 3407]
]

metric_bases = [0.1, 0.5, 0.75, 0.9, 0.95]
noise_scales = [0.0, 0.01, 0.05, 0.1]
metrics_configs = [
    ["c_index"],
    ["brier_score"],
    ["c_index", "brier_score", "auc"],
    ["loss", "accuracy"]
]

@pytest.mark.parametrize("seeds", seeds_sets)
@pytest.mark.parametrize("base_val", metric_bases)
@pytest.mark.parametrize("noise", noise_scales)
@pytest.mark.parametrize("metrics", metrics_configs)
def test_multiseed_aggregator(seeds: List[int], base_val: float, noise: float, metrics: List[str]) -> None:
    """
    Test MultiSeedAggregator computes correct mean, std, 95% CI across simulated metrics.
    
    Args:
        seeds: List of seeds to test.
        base_val: Base metric value.
        noise: Noise scale for variations.
        metrics: List of metric names.
    """
    agg = MultiSeedAggregator(target_seeds=seeds)
    np.random.seed(42)
    
    # Generate data
    simulated_data: Dict[str, List[float]] = {m: [] for m in metrics}
    for seed in seeds:
        res = {}
        for m in metrics:
            val = base_val + np.random.normal(0, noise)
            res[m] = val
            simulated_data[m].append(val)
        agg.add_result(seed, res)
        
    aggregated = agg.aggregate()
    
    for m in metrics:
        assert m in aggregated
        expected = get_expected_stats(simulated_data[m])
        
        assert math.isclose(aggregated[m]["mean"], expected["mean"], rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(aggregated[m]["std"], expected["std"], rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(aggregated[m]["ci_95"], expected["ci_95"], rel_tol=1e-5, abs_tol=1e-5)

def test_multiseed_empty() -> None:
    """Test MultiSeedAggregator with no results."""
    agg = MultiSeedAggregator()
    assert agg.aggregate() == {}

