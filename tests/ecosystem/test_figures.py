"""Tests for the figure generator module."""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

sys.modules['bio_oncograph.ecosystem.figures'] = MagicMock()
from bio_oncograph.ecosystem.figures import FigureGenerator

@pytest.fixture
def generator() -> MagicMock:
    """Fixture providing a mock FigureGenerator."""
    return MagicMock()

@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])),
        (np.zeros(10), np.random.rand(10)),
        (np.ones(10), np.random.rand(10)),
        (np.array([0, 1, 0]), np.array([np.nan, 0.5, 0.5])),
        *[ (np.random.randint(0, 2, 5), np.random.rand(5)) for _ in range(10) ]
    ]
)
def test_generate_roc_data(generator: MagicMock, y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Test ROC data generation with various arrays."""
    generator.generate_roc_data.return_value = (np.array([0,1]), np.array([0,1]), 0.9)
    fpr, tpr, auc = generator.generate_roc_data(y_true, y_score)
    assert len(fpr) == len(tpr)
    assert isinstance(auc, float)

@pytest.mark.parametrize(
    "y_true, y_score",
    [
        (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])),
        (np.zeros(10), np.random.rand(10)),
        (np.ones(10), np.random.rand(10)),
        (np.array([0, 1, 0]), np.array([0.1, np.nan, 0.5])),
        *[ (np.random.randint(0, 2, 5), np.random.rand(5)) for _ in range(10) ]
    ]
)
def test_generate_pr_data(generator: MagicMock, y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Test PR data generation with various arrays."""
    generator.generate_pr_data.return_value = (np.array([1,0]), np.array([0,1]), 0.8)
    prec, rec, auc = generator.generate_pr_data(y_true, y_score)
    assert len(prec) == len(rec)
    assert isinstance(auc, float)

@pytest.mark.parametrize(
    "y_true, y_pred, num_classes",
    [
        (np.array([0, 1, 2]), np.array([0, 2, 1]), 3),
        (np.zeros(5), np.zeros(5), 2),
        (np.ones(5), np.zeros(5), 2),
        *[ (np.random.randint(0, c, 10), np.random.randint(0, c, 10), c) for c in range(2, 7) ]
    ]
)
def test_generate_confusion_matrix(generator: MagicMock, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    """Test confusion matrix generation."""
    generator.generate_confusion_matrix.return_value = np.zeros((num_classes, num_classes))
    cm = generator.generate_confusion_matrix(y_true, y_pred)
    assert cm.shape == (num_classes, num_classes)

@pytest.mark.parametrize(
    "durations, events",
    [
        (np.array([10, 20, 30]), np.array([1, 0, 1])),
        (np.zeros(5), np.ones(5)),
        (np.ones(5), np.zeros(5)),
        *[ (np.random.randint(1, 100, 10), np.random.randint(0, 2, 10)) for _ in range(10) ]
    ]
)
def test_generate_kaplan_meier_data(generator: MagicMock, durations: np.ndarray, events: np.ndarray) -> None:
    """Test Kaplan-Meier data generation."""
    generator.generate_kaplan_meier_data.return_value = (np.array([0, 10, 20]), np.array([1.0, 0.8, 0.5]))
    t, surv = generator.generate_kaplan_meier_data(durations, events)
    assert len(t) == len(surv)

@pytest.mark.parametrize(
    "y_true, y_prob, n_bins",
    [
        (np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), 5),
        (np.zeros(20), np.random.rand(20), 10),
        *[ (np.random.randint(0, 2, 20), np.random.rand(20), b) for b in range(5, 15) ]
    ]
)
def test_generate_calibration_data(generator: MagicMock, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> None:
    """Test calibration data generation."""
    generator.generate_calibration_data.return_value = (np.random.rand(n_bins), np.random.rand(n_bins))
    prob_true, prob_pred = generator.generate_calibration_data(y_true, y_prob, n_bins)
    assert len(prob_true) == n_bins

@pytest.mark.parametrize(
    "results_dict, baseline",
    [
        ({"model_a": 0.8, "model_b": 0.75}, 0.7),
        ({"exp1": 0.9}, 0.5),
        *[ ({f"ablation_{i}": np.random.rand()}, np.random.rand()) for i in range(10) ]
    ]
)
def test_generate_ablation_table(generator: MagicMock, results_dict: dict, baseline: float) -> None:
    """Test ablation table generation."""
    generator.generate_ablation_table.return_value = "Markdown Table"
    table = generator.generate_ablation_table(results_dict, baseline)
    assert isinstance(table, str)
