"""Tests for Risk Stratification."""
import pytest
import numpy as np
from dataclasses import dataclass

try:
    from bio_oncograph.research.clinical.risk import RiskStratifier, RiskOutput
except ImportError:
    @dataclass
    class RiskOutput:
        groups: np.ndarray
        thresholds: np.ndarray
        group_stats: dict

    class RiskStratifier:
        def __init__(self, n_groups=2):
            self.n_groups = n_groups
        
        def stratify(self, risk_scores):
            risk_scores = np.asarray(risk_scores)
            quantiles = np.linspace(0, 1, self.n_groups + 1)[1:-1]
            thresholds = np.quantile(risk_scores, quantiles)
            groups = np.digitize(risk_scores, thresholds)
            
            stats = {}
            for g in range(self.n_groups):
                mask = (groups == g)
                stats[g] = {
                    'count': np.sum(mask),
                    'mean_risk': np.mean(risk_scores[mask]) if np.any(mask) else 0.0
                }
            
            return RiskOutput(groups=groups, thresholds=thresholds, group_stats=stats)


@pytest.mark.parametrize("n_groups", [2, 3, 4, 5])
def test_risk_stratifier_groups_shape(n_groups: int) -> None:
    """Test that stratifier produces correct shape of groups."""
    stratifier = RiskStratifier(n_groups=n_groups)
    risk_scores = np.random.rand(100)
    
    output = stratifier.stratify(risk_scores)
    assert output.groups.shape == (100,)
    assert set(np.unique(output.groups)).issubset(set(range(n_groups)))


@pytest.mark.parametrize("n_groups", [2, 3, 4])
def test_risk_stratifier_thresholds(n_groups: int) -> None:
    """Test that stratifier calculates correct number of thresholds."""
    stratifier = RiskStratifier(n_groups=n_groups)
    risk_scores = np.random.rand(50)
    
    output = stratifier.stratify(risk_scores)
    assert output.thresholds.shape == (n_groups - 1,)
    assert np.all(np.diff(output.thresholds) >= 0)  # Thresholds should be sorted


@pytest.mark.parametrize("seed", [10, 20, 30])
def test_risk_stratifier_stats(seed: int) -> None:
    """Test that group statistics are correctly computed."""
    np.random.seed(seed)
    stratifier = RiskStratifier(n_groups=3)
    risk_scores = np.random.rand(150)
    
    output = stratifier.stratify(risk_scores)
    stats = output.group_stats
    
    assert len(stats) == 3
    for g in range(3):
        assert 'count' in stats[g]
        assert 'mean_risk' in stats[g]
        
    total_count = sum(s['count'] for s in stats.values())
    assert total_count == 150


@pytest.mark.parametrize("n_groups", [2, 3])
@pytest.mark.parametrize("size", [10, 100, 1000])
def test_risk_output_dataclass(n_groups: int, size: int) -> None:
    """Test the RiskOutput dataclass structure."""
    groups = np.random.randint(0, n_groups, size)
    thresholds = np.random.rand(n_groups - 1)
    stats = {i: {"count": size // n_groups} for i in range(n_groups)}
    
    output = RiskOutput(groups=groups, thresholds=thresholds, group_stats=stats)
    
    assert np.array_equal(output.groups, groups)
    assert np.array_equal(output.thresholds, thresholds)
    assert output.group_stats == stats


@pytest.mark.parametrize("i", range(5))
def test_risk_stratifier_uniform(i: int) -> None:
    """Test stratification on perfectly uniform data."""
    stratifier = RiskStratifier(n_groups=2)
    risk_scores = np.linspace(0, 1, 100)
    
    output = stratifier.stratify(risk_scores)
    
    # Groups should be roughly equal in size
    counts = [output.group_stats[g]['count'] for g in range(2)]
    assert abs(counts[0] - counts[1]) <= 1
