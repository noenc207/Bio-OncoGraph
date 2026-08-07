import pytest

class EvaluationReport:
    """Mock for EvaluationReport."""
    def __init__(self, title: str) -> None:
        self.title = title

class ReportGenerator:
    """Mock for ReportGenerator."""
    def generate(self, report: EvaluationReport) -> str:
        return f"# {report.title}"

@pytest.mark.parametrize("idx", list(range(8)))
def test_evaluation_report_init(idx: int) -> None:
    """Test EvaluationReport initialization."""
    report = EvaluationReport(f"Title {idx}")
    assert report.title == f"Title {idx}"

@pytest.mark.parametrize("idx", list(range(8)))
def test_report_generator_generate(idx: int) -> None:
    """Test generating a report document."""
    gen = ReportGenerator()
    report = EvaluationReport(f"Title {idx}")
    content = gen.generate(report)
    assert f"# Title {idx}" in content
