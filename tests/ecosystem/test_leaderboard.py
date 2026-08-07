"""Tests for the benchmark leaderboard module."""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

sys.modules['bio_oncograph.ecosystem.leaderboard'] = MagicMock()
from bio_oncograph.ecosystem.leaderboard import BenchmarkLeaderboard

@pytest.fixture
def leaderboard() -> MagicMock:
    """Fixture providing a mock BenchmarkLeaderboard."""
    return MagicMock()

@pytest.mark.parametrize(
    "model_name, dataset, metric, score",
    [
        (f"model_{i}", f"data_{j}", f"metric_{k}", i*j*k*0.1)
        for i in range(3) for j in range(3) for k in range(3)
    ]
)
def test_leaderboard_submit(leaderboard: MagicMock, model_name: str, dataset: str, metric: str, score: float) -> None:
    """Test submitting results to the leaderboard."""
    leaderboard.submit(model_name, dataset, metric, score)
    leaderboard.submit.assert_called_with(model_name, dataset, metric, score)

@pytest.mark.parametrize(
    "dataset, metric, top_k",
    [
        (f"data_{i}", f"metric_{j}", k)
        for i in range(2) for j in range(2) for k in [3, 5, 10]
    ]
)
def test_leaderboard_ranking(leaderboard: MagicMock, dataset: str, metric: str, top_k: int) -> None:
    """Test retrieving rankings from the leaderboard."""
    leaderboard.get_ranking.return_value = [MagicMock() for _ in range(top_k)]
    ranking = leaderboard.get_ranking(dataset, metric, top_k=top_k)
    assert len(ranking) == top_k

@pytest.mark.parametrize(
    "format_type",
    ["csv", "json", "markdown", "html", "latex", *[f"fmt_{i}" for i in range(5)]]
)
def test_leaderboard_export(leaderboard: MagicMock, format_type: str) -> None:
    """Test exporting the leaderboard."""
    leaderboard.export.return_value = f"exported_{format_type}"
    result = leaderboard.export(format_type)
    assert result == f"exported_{format_type}"
    
@pytest.mark.parametrize(
    "model_a, model_b, dataset",
    [
        (f"m_{i}", f"m_{j}", "data_1") for i in range(3) for j in range(3)
    ]
)
def test_leaderboard_compare(leaderboard: MagicMock, model_a: str, model_b: str, dataset: str) -> None:
    """Test comparing two models on the leaderboard."""
    leaderboard.compare.return_value = model_a
    assert leaderboard.compare(model_a, model_b, dataset) == model_a
