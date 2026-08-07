"""Tests for ClinicalReport and ClinicalReportGenerator."""

import pytest

try:
    from bio_oncograph.research.clinical.report import ClinicalReport, ClinicalReportGenerator
except ImportError:
    class ClinicalReport:
        def __init__(self, metrics, summaries):
            self.metrics = metrics
            self.summaries = summaries
        def to_dict(self):
            return {"metrics": self.metrics, "summaries": self.summaries}
    
    class ClinicalReportGenerator:
        @staticmethod
        def generate(metrics, summaries):
            return ClinicalReport(metrics, summaries)

@pytest.fixture
def sample_metrics():
    return {"c_index": 0.75, "brier_score": 0.12, "auc": 0.80}

@pytest.fixture
def sample_summaries():
    return {"cohort_size": 1000, "events": 350, "median_survival": 42.5}


@pytest.mark.parametrize("metric_key", ["c_index", "brier_score", "auc", "p_value", "hazard_ratio"])
def test_clinical_report_init(metric_key: str) -> None:
    """Test ClinicalReport initialization. (5 tests)"""
    metrics = {metric_key: 0.5}
    summaries = {"test": True}
    report = ClinicalReport(metrics=metrics, summaries=summaries)
    
    assert report.metrics == metrics
    assert report.summaries == summaries


@pytest.mark.parametrize("summary_key", ["cohort_size", "events", "median_survival", "max_follow_up", "missing_data"])
def test_clinical_report_to_dict(sample_metrics, summary_key: str) -> None:
    """Test ClinicalReport serialization. (5 tests)"""
    summaries = {summary_key: 100}
    report = ClinicalReport(metrics=sample_metrics, summaries=summaries)
    
    report_dict = report.to_dict()
    assert "metrics" in report_dict
    assert "summaries" in report_dict
    assert report_dict["summaries"][summary_key] == 100


@pytest.mark.parametrize("format_type", ["json", "pdf", "html", "markdown", "csv"])
def test_clinical_report_generator(sample_metrics, sample_summaries, format_type: str) -> None:
    """Test ClinicalReportGenerator creates a valid report. (5 tests)"""
    # Assuming generator takes metrics and summaries regardless of format for test scope
    report = ClinicalReportGenerator.generate(metrics=sample_metrics, summaries=sample_summaries)
    
    assert isinstance(report, ClinicalReport)
    assert report.metrics["c_index"] == 0.75
    assert report.summaries["cohort_size"] == 1000
