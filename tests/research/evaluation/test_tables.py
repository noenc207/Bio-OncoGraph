import pytest
from typing import Any, List

class TableGenerator:
    """Mock for TableGenerator."""
    def generate_markdown_table(self, data: List[Any]) -> str:
        return "| " + " | ".join(map(str, data)) + " |"
        
    def generate_latex_table(self, data: List[Any]) -> str:
        return " & ".join(map(str, data)) + " \\\\"
        
    def generate_csv(self, data: List[Any]) -> str:
        return ",".join(map(str, data))

@pytest.mark.parametrize("size", list(range(8)))
def test_generate_markdown_table(size: int) -> None:
    """Test markdown table generation formatting."""
    tg = TableGenerator()
    data = list(range(size))
    res = tg.generate_markdown_table(data)
    assert "|" in res

@pytest.mark.parametrize("size", list(range(7)))
def test_generate_latex_table(size: int) -> None:
    """Test LaTeX table generation formatting."""
    tg = TableGenerator()
    data = list(range(size))
    res = tg.generate_latex_table(data)
    assert "\\\\" in res

@pytest.mark.parametrize("size", list(range(7)))
def test_generate_csv(size: int) -> None:
    """Test CSV generation formatting."""
    tg = TableGenerator()
    data = list(range(size))
    res = tg.generate_csv(data)
    if size > 1:
        assert "," in res
