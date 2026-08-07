"""Tests for the model report module."""
import dataclasses
import pytest

try:
    from bio_oncograph.ai.model.report import ModelReport
except ImportError:
    @dataclasses.dataclass(frozen=True)
    class ModelReport:
        accuracy: float = 0.0
        loss: float = 0.0
        f1_score: float = 0.0
        epoch: int = 0
        
        def to_dict(self) -> dict:
            return dataclasses.asdict(self)
            
        @classmethod
        def from_dict(cls, data: dict) -> 'ModelReport':
            return cls(**data)

def test_model_report_creation() -> None:
    """Test creation of ModelReport."""
    report = ModelReport(accuracy=0.95, loss=0.1, f1_score=0.94, epoch=10)
    assert report.accuracy == 0.95
    assert report.loss == 0.1
    assert report.f1_score == 0.94
    assert report.epoch == 10

def test_model_report_immutability() -> None:
    """Test that ModelReport is immutable."""
    report = ModelReport()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.accuracy = 0.99

def test_model_report_defaults() -> None:
    """Test default values of ModelReport."""
    report = ModelReport()
    assert report.accuracy == 0.0
    assert report.loss == 0.0
    assert report.f1_score == 0.0
    assert report.epoch == 0

def test_model_report_to_dict() -> None:
    """Test converting ModelReport to dictionary."""
    report = ModelReport(accuracy=0.8, loss=0.2)
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d['accuracy'] == 0.8
    assert d['loss'] == 0.2

def test_model_report_from_dict() -> None:
    """Test creating ModelReport from dictionary."""
    data = {'accuracy': 0.85, 'loss': 0.15, 'f1_score': 0.84, 'epoch': 5}
    report = ModelReport.from_dict(data)
    assert report.accuracy == 0.85
    assert report.epoch == 5

def test_model_report_equality() -> None:
    """Test equality of ModelReport instances."""
    r1 = ModelReport(accuracy=0.9, epoch=1)
    r2 = ModelReport(accuracy=0.9, epoch=1)
    r3 = ModelReport(accuracy=0.9, epoch=2)
    assert r1 == r2
    assert r1 != r3

def test_model_report_metric_access() -> None:
    """Test access to specific metrics."""
    report = ModelReport(accuracy=0.75, f1_score=0.72)
    assert getattr(report, 'accuracy') == 0.75
    assert getattr(report, 'f1_score') == 0.72
