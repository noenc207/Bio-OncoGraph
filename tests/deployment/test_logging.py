import pytest
import json
from unittest.mock import Mock
from typing import Dict, Any

class AuditLogger:
    def log_request(self, data: Dict[str, Any]) -> str:
        return json.dumps({"audit_req": data})

@pytest.fixture
def logger() -> AuditLogger:
    return AuditLogger()

@pytest.mark.parametrize("req_id", range(35))
def test_log_request_json_structure(logger: AuditLogger, req_id: int) -> None:
    """Test logging requests and validating logged JSON structure."""
    data = {"request_id": req_id, "status": "OK", "payload": "mock_data"}
    result = logger.log_request(data)
    parsed = json.loads(result)
    assert "audit_req" in parsed
    assert parsed["audit_req"]["request_id"] == req_id
    assert parsed["audit_req"]["status"] == "OK"

@pytest.mark.parametrize("level", ["INFO", "WARN", "ERROR", "DEBUG", "TRACE"])
@pytest.mark.parametrize("comp", ["Auth", "DB", "API"])
def test_log_request_metadata(logger: AuditLogger, level: str, comp: str) -> None:
    """Test combinations of logging levels and components."""
    data = {"level": level, "component": comp}
    result = logger.log_request(data)
    parsed = json.loads(result)
    assert parsed["audit_req"]["level"] == level
    assert parsed["audit_req"]["component"] == comp
