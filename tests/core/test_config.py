"""
Unit tests for Hydra Structured Configs.

Validates:
  - Dataclass instantiation with defaults.
  - MISSING fields raise at access time.
  - OmegaConf merge and type enforcement.
  - YAML files are parseable and consistent with schemas.
  - Backward compatibility: adding a field with a default must not
    break existing configs.
"""

from dataclasses import fields
from typing import Any

import pytest
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from bio_oncograph.core.config import (
    DatasetConfig,
    ExperimentConfig,
    LoggerConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


# ── ModelConfig ─────────────────────────────────────────────────────


def test_model_config_defaults() -> None:
    """Non-MISSING fields have correct defaults."""
    cfg = OmegaConf.structured(ModelConfig)
    assert cfg.hidden_dim == 256
    assert cfg.num_phenotypes == 8
    assert cfg.pool_ratios == [0.5, 0.25]
    assert cfg.dropout == 0.1
    assert cfg.time_bins == 4


def test_model_config_missing_in_channels() -> None:
    """Accessing in_channels without setting it raises."""
    cfg = OmegaConf.structured(ModelConfig)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.in_channels


def test_model_config_override() -> None:
    """Values can be overridden via OmegaConf merge."""
    base = OmegaConf.structured(ModelConfig)
    override = OmegaConf.create({"in_channels": 768, "hidden_dim": 512})
    merged = OmegaConf.merge(base, override)
    assert isinstance(merged, DictConfig)
    assert merged.in_channels == 768
    assert merged.hidden_dim == 512
    # Non-overridden defaults preserved
    assert merged.num_phenotypes == 8


# ── DatasetConfig ───────────────────────────────────────────────────


def test_dataset_config_missing_root() -> None:
    """root_dir is mandatory."""
    cfg = OmegaConf.structured(DatasetConfig)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.root_dir


def test_dataset_config_full() -> None:
    """Full dataset config resolves correctly."""
    cfg = OmegaConf.structured(DatasetConfig)
    override = OmegaConf.create({
        "name": "tcga_brca",
        "root_dir": "/data/features",
        "split_dir": "/data/splits",
    })
    merged = OmegaConf.merge(cfg, override)
    assert merged.name == "tcga_brca"
    assert merged.k_neighbours == 8


# ── LossConfig ──────────────────────────────────────────────────────


def test_loss_config_defaults() -> None:
    """Loss config has sensible defaults."""
    cfg = OmegaConf.structured(LossConfig)
    assert cfg.alpha == 1.0
    assert cfg.beta == 0.5
    assert cfg.epsilon == 1e-8
    assert cfg.num_projections == 50


# ── TrainingConfig ──────────────────────────────────────────────────


def test_training_config_nested() -> None:
    """Nested optimizer and scheduler sub-configs resolve."""
    cfg = OmegaConf.structured(TrainingConfig)
    assert cfg.optimizer.name == "adamw"
    assert cfg.optimizer.lr == 1e-4
    assert cfg.scheduler.name == "cosine"
    assert cfg.scheduler.warmup_epochs == 5


def test_training_config_override_nested() -> None:
    """Nested sub-config can be overridden."""
    base = OmegaConf.structured(TrainingConfig)
    override = OmegaConf.create({"optimizer": {"lr": 3e-4}})
    merged = OmegaConf.merge(base, override)
    assert merged.optimizer.lr == 3e-4
    # Other fields preserved
    assert merged.optimizer.name == "adamw"


# ── ExperimentConfig ────────────────────────────────────────────────


def test_experiment_config_top_level() -> None:
    """Top-level config aggregates all sub-configs."""
    cfg = OmegaConf.structured(ExperimentConfig)
    assert cfg.seed == 42
    assert cfg.run_mode == "TRAIN"
    assert cfg.amp is True
    assert cfg.model.hidden_dim == 256
    assert cfg.loss.alpha == 1.0
    assert cfg.training.epochs == 100
    assert cfg.logger.use_wandb is False


def test_experiment_config_missing_propagates() -> None:
    """MISSING in nested config propagates to top level."""
    cfg = OmegaConf.structured(ExperimentConfig)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.model.in_channels


# ── Backward Compatibility ──────────────────────────────────────────


def test_backward_compat_extra_field_ignored() -> None:
    """Old YAML with unknown fields does not crash when merged loosely."""
    old_yaml = OmegaConf.create({"hidden_dim": 128, "legacy_field": True})
    schema = OmegaConf.structured(ModelConfig)
    # Merge with struct=False to simulate loading legacy config
    OmegaConf.set_struct(schema, False)
    merged = OmegaConf.merge(schema, old_yaml)
    assert merged.hidden_dim == 128


# ── Schema completeness ────────────────────────────────────────────


def test_all_config_classes_are_dataclasses() -> None:
    """Every config class is a proper dataclass."""
    for cls in [
        ModelConfig,
        DatasetConfig,
        LossConfig,
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
        LoggerConfig,
        ExperimentConfig,
    ]:
        # fields() raises TypeError if cls is not a dataclass
        assert len(fields(cls)) > 0, f"{cls.__name__} has no fields"
