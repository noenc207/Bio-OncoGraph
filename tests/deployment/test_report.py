import pytest
from typing import Dict, Any

class DeploymentReport:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

class ReportGenerator:
    def compile(self, state: Dict[str, Any]) -> DeploymentReport:
        return DeploymentReport({"compiled_state": state})

@pytest.fixture
def generator() -> ReportGenerator:
    return ReportGenerator()

@pytest.mark.parametrize("uptime", range(90, 100))
@pytest.mark.parametrize("errors", range(0, 3))
def test_report_compilation(generator: ReportGenerator, uptime: int, errors: int) -> None:
    """Test deployment report generation and mocked system state compilation."""
    mock_state = {"uptime": uptime, "error_count": errors}
    report = generator.compile(mock_state)
    assert report.data["compiled_state"]["uptime"] == uptime
    assert report.data["compiled_state"]["error_count"] == errors
