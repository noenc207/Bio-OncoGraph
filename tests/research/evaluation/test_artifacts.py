import pytest
from typing import Any, Dict

class ArtifactExporter:
    """Mock for ArtifactExporter."""
    def export_json(self, path: str, data: Dict[str, Any]) -> bool:
        return True
        
    def export_text(self, path: str, text: str) -> bool:
        return True

@pytest.mark.parametrize("idx", list(range(8)))
def test_export_json(idx: int) -> None:
    """Test exporting data to JSON mock."""
    exp = ArtifactExporter()
    assert exp.export_json(f"path_{idx}.json", {"id": idx})

@pytest.mark.parametrize("idx", list(range(8)))
def test_export_text(idx: int) -> None:
    """Test exporting text to a file mock."""
    exp = ArtifactExporter()
    assert exp.export_text(f"path_{idx}.txt", f"text content {idx}")
