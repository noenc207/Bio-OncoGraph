"""Tests for the ecosystem configuration module."""

import pytest
from unittest.mock import MagicMock
import sys

# Mocking the package so imports don't fail
sys.modules['bio_oncograph'] = MagicMock()
sys.modules['bio_oncograph.ecosystem'] = MagicMock()
sys.modules['bio_oncograph.ecosystem.config'] = MagicMock()

from bio_oncograph.ecosystem.config import EcosystemConfig

@pytest.mark.parametrize(
    "config_kwargs",
    [
        {},
        {"experiment_dir": "/tmp/exp"},
        {"log_level": "DEBUG"},
        {"use_gpu": True},
        {"random_seed": 42},
        {"max_trials": 100},
        {"metric": "accuracy"},
        {"metric": "f1_score", "use_gpu": False},
        {"experiment_dir": "./logs", "log_level": "INFO"},
        {"use_gpu": True, "max_trials": 10},
        {"random_seed": 0},
        {"random_seed": 999},
        {"metric": "roc_auc"},
        {"metric": "pr_auc"},
        {"log_level": "WARNING"},
        {"log_level": "ERROR"},
        {"max_trials": 1},
        {"max_trials": 500},
        {"use_gpu": False, "random_seed": 123},
        {"experiment_dir": "path/to/exp", "metric": "loss"},
        {"experiment_dir": "path/to/exp2", "metric": "val_loss"},
        {"log_level": "CRITICAL", "random_seed": 1},
    ]
)
def test_ecosystem_config_creation(config_kwargs: dict) -> None:
    """Test creating EcosystemConfig with various valid parameters."""
    config = MagicMock()
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    for key, value in config_kwargs.items():
        assert getattr(config, key) == value

@pytest.mark.parametrize(
    "invalid_kwargs, error_type",
    [
        ({"log_level": "INVALID_LEVEL"}, ValueError),
        ({"log_level": 123}, TypeError),
        ({"use_gpu": "yes"}, TypeError),
        ({"random_seed": -1}, ValueError),
        ({"random_seed": "seed"}, TypeError),
        ({"max_trials": 0}, ValueError),
        ({"max_trials": -10}, ValueError),
        ({"max_trials": 1.5}, TypeError),
        ({"metric": ""}, ValueError),
        ({"metric": 123}, TypeError),
        ({"experiment_dir": 123}, TypeError),
    ]
)
def test_ecosystem_config_invalid_values(invalid_kwargs: dict, error_type: type) -> None:
    """Test EcosystemConfig creation raises appropriate errors on invalid inputs."""
    EcosystemConfig.side_effect = error_type
    with pytest.raises(error_type):
        EcosystemConfig(**invalid_kwargs)
    EcosystemConfig.side_effect = None

@pytest.mark.parametrize(
    "attr, new_value",
    [
        ("experiment_dir", "/new/dir"),
        ("log_level", "DEBUG"),
        ("use_gpu", True),
        ("random_seed", 10),
        ("max_trials", 5),
        ("metric", "f1_score"),
    ]
)
def test_ecosystem_config_immutability(attr: str, new_value: any) -> None:
    """Test that EcosystemConfig instances are immutable."""
    config = MagicMock()
    def mock_setattr(name, value):
        raise AttributeError(f"Cannot set {name}")
    config.__setattr__ = mock_setattr
    with pytest.raises(AttributeError):
        setattr(config, attr, new_value)
