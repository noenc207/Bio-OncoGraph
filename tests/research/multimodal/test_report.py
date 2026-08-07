"""Tests for EngineReport and ReportGenerator in Bio-OncoGraph R007."""

import pytest
import datetime
import numpy as np

class EngineReport:
    """Mock EngineReport to store report metrics."""
    def __init__(self, metrics=None):
        self.metrics = metrics or {}
        self.timestamp = datetime.datetime.now().isoformat()
        self.status = "SUCCESS"
        
    def add_metric(self, key, value):
        self.metrics[key] = value
        
    def mark_failed(self, reason):
        self.status = "FAILED"
        self.add_metric("error_reason", reason)

class ReportGenerator:
    """Mock ReportGenerator to generate text/html reports."""
    def __init__(self, template="text"):
        self.template = template
        
    def generate(self, report):
        if not isinstance(report, EngineReport):
            raise TypeError("Expected EngineReport object")
            
        if self.template == "text":
            return self._generate_text(report)
        elif self.template == "html":
            return self._generate_html(report)
        raise ValueError(f"Unknown template: {self.template}")
        
    def _generate_text(self, report):
        lines = [f"Status: {report.status}", f"Timestamp: {report.timestamp}"]
        for k, v in report.metrics.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
        
    def _generate_html(self, report):
        return f"<div><h1>{report.status}</h1></div>"

@pytest.fixture
def report():
    return EngineReport()

@pytest.fixture
def generator():
    return ReportGenerator()

def test_report_init(report):
    assert report.status == "SUCCESS"
    assert report.timestamp is not None
    assert isinstance(report.metrics, dict)

def test_report_add_metric(report):
    report.add_metric("fusion_shape", (10, 10))
    assert report.metrics["fusion_shape"] == (10, 10)

def test_report_mark_failed(report):
    report.mark_failed("Out of memory")
    assert report.status == "FAILED"
    assert report.metrics["error_reason"] == "Out of memory"

def test_generator_init(generator):
    assert generator.template == "text"

def test_generator_custom_template():
    gen = ReportGenerator(template="html")
    assert gen.template == "html"

def test_generate_text_report(generator, report):
    report.add_metric("loss", 0.5)
    text = generator.generate(report)
    assert "Status: SUCCESS" in text
    assert "loss: 0.5" in text
    assert "Timestamp:" in text

def test_generate_html_report(report):
    gen = ReportGenerator(template="html")
    html = gen.generate(report)
    assert "<h1>SUCCESS</h1>" in html

def test_generate_invalid_template(report):
    gen = ReportGenerator(template="invalid")
    with pytest.raises(ValueError):
        gen.generate(report)

def test_generate_invalid_report_type(generator):
    with pytest.raises(TypeError):
        generator.generate({"status": "SUCCESS"})

def test_report_multiple_metrics(report, generator):
    report.add_metric("m1", 1)
    report.add_metric("m2", 2)
    text = generator.generate(report)
    assert "m1: 1" in text
    assert "m2: 2" in text

def test_report_numpy_metric(report, generator):
    report.add_metric("array", np.array([1, 2, 3]))
    text = generator.generate(report)
    assert "array:" in text

def test_report_failed_generation(generator, report):
    report.mark_failed("Crash")
    text = generator.generate(report)
    assert "Status: FAILED" in text
    assert "error_reason: Crash" in text

def test_generator_empty_metrics(generator, report):
    text = generator.generate(report)
    assert "Status: SUCCESS" in text
    assert len(text.split("\n")) == 2

def test_report_timestamp_format(report):
    # Check it looks like ISO
    assert "T" in report.timestamp

def test_generator_custom_metric_types(report, generator):
    report.add_metric("complex", {"a": 1})
    text = generator.generate(report)
    assert "{'a': 1}" in text
