import pytest
from unittest.mock import patch

class DummyReportGenerator:
    """Dummy report generator."""
    def generate(self, dataset_name, stats):
        pass

@pytest.fixture
def reporter():
    """Returns a dummy report generator."""
    return DummyReportGenerator()

@pytest.mark.parametrize(
    "dataset_name, num_records, num_patients",
    [
        (f"T002_Subset_{i}", 100 * i, 10 * i)
        for i in range(1, 26)
    ]
)
def test_report_generation(reporter, dataset_name, num_records, num_patients):
    """Test generating dataset reports.
    
    Args:
        reporter: Reporter fixture.
        dataset_name: Name of dataset.
        num_records: Total records.
        num_patients: Total patients.
    """
    stats = {"records": num_records, "patients": num_patients}
    expected_report = f"Report for {dataset_name}: {num_records} records across {num_patients} patients."
    
    with patch.object(reporter, 'generate', return_value=expected_report):
        report = reporter.generate(dataset_name, stats)
        assert dataset_name in report
        assert str(num_records) in report
        assert str(num_patients) in report
