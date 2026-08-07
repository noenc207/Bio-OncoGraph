"""Tests for the experiment management module."""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

sys.modules['bio_oncograph.ecosystem.experiments'] = MagicMock()
from bio_oncograph.ecosystem.experiments import ExperimentManager, Experiment

@pytest.fixture
def manager() -> MagicMock:
    """Fixture providing a mock ExperimentManager."""
    return MagicMock()

@pytest.mark.parametrize(
    "exp_name, tags, description",
    [
        (f"exp_{i}", [f"tag_{i}"], f"Desc {i}") for i in range(15)
    ]
)
def test_create_experiment(manager: MagicMock, exp_name: str, tags: list[str], description: str) -> None:
    """Test creating an experiment with different attributes."""
    exp = MagicMock()
    exp.name = exp_name
    exp.tags = tags
    exp.description = description
    manager.create_experiment.return_value = exp
    
    result = manager.create_experiment(name=exp_name, tags=tags, description=description)
    assert result.name == exp_name
    assert result.tags == tags
    assert result.description == description

@pytest.mark.parametrize(
    "exp_name, metric_name, metric_value, step",
    [
        (f"exp_{i}", f"metric_{j}", i * j * 0.1, j) 
        for i in range(5) for j in range(3)
    ]
)
def test_log_metric(manager: MagicMock, exp_name: str, metric_name: str, metric_value: float, step: int) -> None:
    """Test logging metrics to experiments."""
    manager.log_metric(exp_name, metric_name, metric_value, step)
    manager.log_metric.assert_called_with(exp_name, metric_name, metric_value, step)

@pytest.mark.parametrize(
    "query_tags, expected_count",
    [
        ([f"tag_{i}"], i % 3) for i in range(10)
    ]
)
def test_list_experiments(manager: MagicMock, query_tags: list[str], expected_count: int) -> None:
    """Test listing experiments by tags."""
    manager.list_experiments.return_value = [MagicMock() for _ in range(expected_count)]
    results = manager.list_experiments(tags=query_tags)
    assert len(results) == expected_count

@pytest.mark.parametrize(
    "exp1_name, exp2_name, metric, expected_winner",
    [
        (f"exp_{i}", f"exp_{i+1}", "acc", f"exp_{i}") for i in range(10)
    ]
)
def test_compare_experiments(manager: MagicMock, exp1_name: str, exp2_name: str, metric: str, expected_winner: str) -> None:
    """Test comparing two experiments based on a metric."""
    manager.compare.return_value = expected_winner
    winner = manager.compare(exp1_name, exp2_name, metric)
    assert winner == expected_winner

def test_experiment_get(manager: MagicMock) -> None:
    """Test retrieving an experiment."""
    manager.get.return_value = MagicMock(name="exp_1")
    assert manager.get("exp_1").name == "exp_1"
