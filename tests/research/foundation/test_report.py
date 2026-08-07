"""Tests for Graph Foundation Report."""

import pytest
from dataclasses import dataclass, asdict
import json
import os

@dataclass
class TrainingReport:
    model_name: str
    total_epochs: int
    final_loss: float
    metrics: dict

class ReportGenerator:
    def __init__(self, title: str):
        self.title = title

    def generate(self, report: TrainingReport) -> str:
        return f"# {self.title}\nModel: {report.model_name}\nLoss: {report.final_loss}"
    
    def save(self, report: TrainingReport, path: str):
        with open(path, 'w') as f:
            f.write(self.generate(report))

@pytest.fixture
def sample_report():
    return TrainingReport(model_name="GraphMAE", total_epochs=100, final_loss=0.05, metrics={"acc": 0.95})

@pytest.fixture
def generator():
    return ReportGenerator("Test Report")

def test_report_dataclass_init(sample_report):
    assert sample_report.model_name == "GraphMAE"
    assert sample_report.total_epochs == 100

def test_report_dataclass_metrics(sample_report):
    assert "acc" in sample_report.metrics
    assert sample_report.metrics["acc"] == 0.95

def test_generator_init(generator):
    assert generator.title == "Test Report"

def test_generator_generate(generator, sample_report):
    out = generator.generate(sample_report)
    assert "Test Report" in out
    assert "GraphMAE" in out
    assert "0.05" in out

def test_generator_save(generator, sample_report, tmp_path):
    p = tmp_path / "report.md"
    generator.save(sample_report, str(p))
    assert p.exists()
    assert "GraphMAE" in p.read_text()

def test_report_asdict(sample_report):
    d = asdict(sample_report)
    assert d["model_name"] == "GraphMAE"
    assert d["total_epochs"] == 100

def test_generator_empty_metrics():
    r = TrainingReport("M", 1, 1.0, {})
    g = ReportGenerator("T")
    out = g.generate(r)
    assert "M" in out

def test_report_negative_loss():
    r = TrainingReport("M", 1, -0.5, {})
    assert r.final_loss == -0.5

def test_report_zero_epochs():
    r = TrainingReport("M", 0, 1.0, {})
    assert r.total_epochs == 0

def test_generator_custom_title():
    g = ReportGenerator("Custom")
    assert g.title == "Custom"

def test_generator_generate_format():
    r = TrainingReport("M", 1, 1.0, {})
    g = ReportGenerator("T")
    out = g.generate(r)
    assert out.startswith("# T")

def test_report_large_metrics():
    m = {f"k_{i}": i for i in range(100)}
    r = TrainingReport("M", 1, 1.0, m)
    assert len(r.metrics) == 100

def test_generator_save_invalid_path(generator, sample_report):
    with pytest.raises(Exception):
        generator.save(sample_report, "/invalid/path/report.md")

def test_report_equality(sample_report):
    r2 = TrainingReport("GraphMAE", 100, 0.05, {"acc": 0.95})
    assert sample_report == r2

def test_report_inequality(sample_report):
    r2 = TrainingReport("GraphMAE", 101, 0.05, {"acc": 0.95})
    assert sample_report != r2

def test_report_type_annotations():
    assert TrainingReport.__annotations__["final_loss"] == float
    assert TrainingReport.__annotations__["total_epochs"] == int

def test_generator_generate_type(generator, sample_report):
    assert isinstance(generator.generate(sample_report), str)

def test_report_model_name_empty():
    r = TrainingReport("", 1, 1.0, {})
    assert r.model_name == ""

def test_generator_overwrite_file(generator, sample_report, tmp_path):
    p = tmp_path / "report.md"
    generator.save(sample_report, str(p))
    r2 = TrainingReport("Model2", 1, 0.1, {})
    generator.save(r2, str(p))
    assert "Model2" in p.read_text()
    assert "GraphMAE" not in p.read_text()

def test_report_metrics_mutation(sample_report):
    sample_report.metrics["new"] = 1.0
    assert sample_report.metrics["new"] == 1.0
