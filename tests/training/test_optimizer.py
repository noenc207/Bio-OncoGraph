"""Tests for OptimizerFactory parameter grouping."""

from typing import Any, Dict, Iterator
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW


class DummyModel(nn.Module):
    """Dummy model with weight decay and no-weight decay parameters."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, bias=True)
        self.bn = nn.BatchNorm2d(16)
        self.linear = nn.Linear(16, 10, bias=True)


class OptimizerFactory:
    """Mock implementation for testing."""
    @staticmethod
    def create(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        weight_decay = config.get("weight_decay", 0.0)
        no_decay = ["bias", "bn"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if config["type"] == "adamw":
            return AdamW(optimizer_grouped_parameters, lr=config["lr"])
        if config["type"] == "sgd":
            return SGD(optimizer_grouped_parameters, lr=config["lr"], momentum=config.get("momentum", 0.9))
        raise ValueError("Unknown optimizer type")


@pytest.fixture
def model() -> DummyModel:
    """Provides a dummy model."""
    return DummyModel()


@pytest.mark.parametrize("lr", [1e-3, 1e-4, 5e-5])
@pytest.mark.parametrize("weight_decay", [0.01, 1e-4, 0.0])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.9, 0.99)])
def test_create_adamw(model: DummyModel, lr: float, weight_decay: float, betas: tuple) -> None:
    """Test AdamW creation and parameter grouping."""
    config = {"type": "adamw", "lr": lr, "weight_decay": weight_decay, "betas": betas}
    opt = OptimizerFactory.create(model, config)
    assert isinstance(opt, AdamW)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == weight_decay
    assert opt.param_groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize("lr", [0.1, 0.01])
@pytest.mark.parametrize("momentum", [0.9, 0.99])
@pytest.mark.parametrize("weight_decay", [1e-4, 1e-5])
def test_create_sgd(model: DummyModel, lr: float, momentum: float, weight_decay: float) -> None:
    """Test SGD creation and parameter grouping."""
    config = {"type": "sgd", "lr": lr, "momentum": momentum, "weight_decay": weight_decay}
    opt = OptimizerFactory.create(model, config)
    assert isinstance(opt, SGD)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == weight_decay
    assert opt.param_groups[1]["weight_decay"] == 0.0


@pytest.mark.parametrize("invalid_type", ["rmsprop", "adagrad", "unknown"])
def test_create_invalid_optimizer(model: DummyModel, invalid_type: str) -> None:
    """Test factory raises error on invalid type."""
    config = {"type": invalid_type, "lr": 1e-3}
    with pytest.raises(ValueError):
        OptimizerFactory.create(model, config)
