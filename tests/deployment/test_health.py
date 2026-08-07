import pytest
from typing import Dict

class HealthChecker:
    def check_registry(self, registry: Dict[str, bool]) -> bool:
        return all(registry.values()) if registry else False

    def check_memory_threshold(self, memory_used: int, threshold: int) -> bool:
        return memory_used <= threshold

@pytest.fixture
def checker() -> HealthChecker:
    return HealthChecker()

@pytest.mark.parametrize("registry, expected", [
    ({"service_a": True, "service_b": True}, True),
    ({"service_a": True, "service_b": False}, False),
    ({}, False)
] * 12)
def test_check_registry(checker: HealthChecker, registry: Dict[str, bool], expected: bool) -> None:
    """Test registry and deployment status checks."""
    assert checker.check_registry(registry) == expected

@pytest.mark.parametrize("memory_used", range(0, 100, 5))
@pytest.mark.parametrize("threshold", [50, 75])
def test_check_memory_threshold(checker: HealthChecker, memory_used: int, threshold: int) -> None:
    """Test memory threshold breaches."""
    expected_status = memory_used <= threshold
    assert checker.check_memory_threshold(memory_used, threshold) == expected_status
