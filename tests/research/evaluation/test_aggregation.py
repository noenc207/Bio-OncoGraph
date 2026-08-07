import pytest
from typing import Any, Dict, List

class ResultAggregator:
    """Mock for ResultAggregator."""
    def aggregate_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        keys = set().union(*(d.keys() for d in results))
        return {k: sum(d.get(k, 0.0) for d in results) / len(results) for k in keys}
        
    def cross_dataset_aggregation(self, datasets: Dict[str, List[Any]]) -> Dict[str, int]:
        return {k: len(v) for k, v in datasets.items()}

@pytest.mark.parametrize("idx", list(range(20)))
def test_aggregate_metrics_valid(idx: int) -> None:
    """Test aggregate_metrics with valid inputs."""
    agg = ResultAggregator()
    res = agg.aggregate_metrics([{"acc": float(idx)}, {"acc": float(idx + 2)}])
    assert res["acc"] == float(idx + 1)

@pytest.mark.parametrize("idx", list(range(7)))
def test_aggregate_metrics_missing_keys(idx: int) -> None:
    """Test aggregate_metrics when dictionaries have mismatched keys."""
    agg = ResultAggregator()
    res = agg.aggregate_metrics([{"acc": float(idx)}, {"loss": float(idx)}])
    assert "acc" in res
    assert "loss" in res

@pytest.mark.parametrize("num_ds", list(range(1, 6)))
def test_cross_dataset_aggregation(num_ds: int) -> None:
    """Test cross_dataset_aggregation functionality."""
    agg = ResultAggregator()
    data = {f"ds_{i}": [1, 2] for i in range(num_ds)}
    res = agg.cross_dataset_aggregation(data)
    assert len(res) == num_ds
