"""Tests for evaluation configuration module."""

import pytest
from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationConfig:
    """Mock configuration for evaluation."""
    seed: int = 42
    cv_folds: int = 5
    metrics: tuple = ('accuracy', 'f1', 'roc_auc')
    statistical_tests: tuple = ('ttest', 'wilcoxon')
    alpha: float = 0.05
    n_bootstraps: int = 1000

    def __post_init__(self):
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1")

def test_config_defaults():
    """Test default values of EvaluationConfig."""
    config = EvaluationConfig()
    assert config.seed == 42
    assert config.cv_folds == 5
    assert config.alpha == 0.05
    assert config.n_bootstraps == 1000

def test_config_immutability():
    """Test that EvaluationConfig is frozen."""
    config = EvaluationConfig()
    with pytest.raises(Exception):
        config.seed = 99

@pytest.mark.parametrize("seed", [0, 42, 999, -1, 12345])
def test_config_seed_valid(seed):
    """Test valid seed values."""
    config = EvaluationConfig(seed=seed)
    assert config.seed == seed

@pytest.mark.parametrize("cv_folds", [2, 3, 5, 10, 20])
def test_config_cv_folds_valid(cv_folds):
    """Test valid cross-validation folds."""
    config = EvaluationConfig(cv_folds=cv_folds)
    assert config.cv_folds == cv_folds

@pytest.mark.parametrize("cv_folds", [0, 1, -1, -5])
def test_config_cv_folds_invalid(cv_folds):
    """Test invalid cross-validation folds."""
    with pytest.raises(ValueError, match="cv_folds must be >= 2"):
        EvaluationConfig(cv_folds=cv_folds)

@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1, 0.5, 0.99])
def test_config_alpha_valid(alpha):
    """Test valid significance levels."""
    config = EvaluationConfig(alpha=alpha)
    assert config.alpha == alpha

@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5, 2.0])
def test_config_alpha_invalid(alpha):
    """Test invalid significance levels."""
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        EvaluationConfig(alpha=alpha)

@pytest.mark.parametrize("n_bootstraps", [1, 10, 100, 1000, 10000])
def test_config_n_bootstraps_valid(n_bootstraps):
    """Test valid number of bootstraps."""
    config = EvaluationConfig(n_bootstraps=n_bootstraps)
    assert config.n_bootstraps == n_bootstraps

@pytest.mark.parametrize("n_bootstraps", [0, -1, -100])
def test_config_n_bootstraps_invalid(n_bootstraps):
    """Test invalid number of bootstraps."""
    with pytest.raises(ValueError, match="n_bootstraps must be >= 1"):
        EvaluationConfig(n_bootstraps=n_bootstraps)

@pytest.mark.parametrize("metrics", [
    ('accuracy',),
    ('accuracy', 'f1'),
    ('accuracy', 'precision', 'recall', 'f1'),
    tuple()
])
def test_config_metrics_valid(metrics):
    """Test valid metrics configurations."""
    config = EvaluationConfig(metrics=metrics)
    assert config.metrics == metrics

@pytest.mark.parametrize("tests", [
    ('ttest',),
    ('wilcoxon', 'ttest'),
    tuple()
])
def test_config_statistical_tests_valid(tests):
    """Test valid statistical tests configurations."""
    config = EvaluationConfig(statistical_tests=tests)
    assert config.statistical_tests == tests
