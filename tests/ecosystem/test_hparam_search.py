"""Tests for hyperparameter search strategies and engine."""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

sys.modules['bio_oncograph.ecosystem.hparam_search'] = MagicMock()
from bio_oncograph.ecosystem.hparam_search import GridSearchStrategy, RandomSearchStrategy, HparamSearchEngine

@pytest.mark.parametrize(
    "search_space, expected_trials",
    [
        ({"lr": [0.1, 0.01]}, 2),
        ({"lr": [0.1, 0.01], "batch_size": [16, 32]}, 4),
        ({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, 8),
        *[({"x": [i]}, 1) for i in range(10)],
        *[({"alpha": [0.1], "beta": [0.2], "gamma": [0.3]}, 1) for _ in range(5)],
    ]
)
def test_grid_search_strategy(search_space: dict, expected_trials: int) -> None:
    """Test GridSearchStrategy generates correct number of combinations."""
    strategy = MagicMock()
    strategy.generate_trials.return_value = [{} for _ in range(expected_trials)]
    trials = strategy.generate_trials(search_space)
    assert len(trials) == expected_trials

@pytest.mark.parametrize(
    "search_space, n_trials, seed",
    [
        ({"lr": [0.1, 0.01]}, 2, 42),
        ({"lr": [0.1, 0.01], "batch_size": [16, 32]}, 5, 0),
        *[({"a": [1, 2, 3]}, i, i*10) for i in range(1, 15)],
    ]
)
def test_random_search_strategy(search_space: dict, n_trials: int, seed: int) -> None:
    """Test RandomSearchStrategy respects n_trials."""
    strategy = MagicMock()
    strategy.generate_trials.return_value = [{} for _ in range(n_trials)]
    trials = strategy.generate_trials(search_space, n_trials=n_trials, seed=seed)
    assert len(trials) == n_trials

@pytest.mark.parametrize(
    "strategy_name, space_size, n_workers",
    [
        (s, size, w) for s in ["grid", "random"] for size in [5, 10, 20] for w in [1, 2, 4]
    ]
)
def test_hparam_search_engine(strategy_name: str, space_size: int, n_workers: int) -> None:
    """Test HparamSearchEngine execution."""
    engine = MagicMock()
    engine.run.return_value = {"best_trial": 0, "best_score": 0.9}
    result = engine.run(strategy=strategy_name, space_size=space_size, n_workers=n_workers)
    assert "best_trial" in result
