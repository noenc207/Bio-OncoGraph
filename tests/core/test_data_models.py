"""
Unit tests for core data models and typing definitions.
"""

import pytest
import torch
from bio_oncograph.core.data_models import SurvivalTarget, BioGraphOutput, BioGraphData
from bio_oncograph.core.typing import MetricsDict, LossOutput, PoolingLog


def test_metrics_dict_typing() -> None:
    """Test TypedDict instantiation."""
    metrics: MetricsDict = {
        "c_index": 0.85,
        "brier_score": 0.12,
        "d_calibration": 0.05
    }
    assert metrics["c_index"] == 0.85


def test_survival_target() -> None:
    """Test SurvivalTarget dataclass instantiation."""
    time_bins = torch.tensor([1, 2, 3])
    events = torch.tensor([True, False, True])
    target = SurvivalTarget(time_bins=time_bins, event_indicators=events)
    
    assert torch.equal(target.time_bins, time_bins)
    assert torch.equal(target.event_indicators, events)


def test_biograph_output() -> None:
    """Test BioGraphOutput dataclass."""
    logits = torch.randn(2, 5)
    curves = torch.rand(2, 5)
    output = BioGraphOutput(hazard_logits=logits, survival_curves=curves)
    
    assert output.persistence_diagrams is None
    assert output.attention_weights is None
    assert torch.equal(output.hazard_logits, logits)


def test_biograph_data() -> None:
    """Test BioGraphData PyG wrapper."""
    x = torch.randn(10, 64)
    pos = torch.randn(10, 2)
    edge_index = torch.randint(0, 10, (2, 20))
    
    target = SurvivalTarget(time_bins=torch.tensor([1]), event_indicators=torch.tensor([True]))
    
    data = BioGraphData(x=x, pos=pos, edge_index=edge_index, survival_target=target)
    
    assert torch.equal(data.x, x)
    assert torch.equal(data.pos, pos)
    assert data.survival_target is not None
    assert data.survival_target.time_bins.item() == 1
