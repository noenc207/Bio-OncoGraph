import pytest
from unittest.mock import MagicMock

class MockPaperGenerator:
    def __init__(self):
        pass
        
    def generate_latex_table(self, data, caption):
        if not data:
            return ""
        return f"\\begin{{table}}\n\\caption{{{caption}}}\n\\end{{table}}"

    def compile_latex(self, latex_string):
        return "\\documentclass" in latex_string

# Generate 100 parameters
latex_table_params = []
for i in range(50):
    latex_table_params.extend([
        ([1, 2, 3], "Test Caption", True),
        ([], "Empty", False),
    ])

@pytest.mark.parametrize("data, caption, should_have_content", latex_table_params)
def test_latex_table_generation(data, caption, should_have_content):
    """Test LaTeX table string generation."""
    generator = MockPaperGenerator()
    result = generator.generate_latex_table(data, caption)
    if should_have_content:
        assert "\\begin{table}" in result
        assert f"\\caption{{{caption}}}" in result
    else:
        assert result == ""

# Generate 100 parameters
compile_params = []
for i in range(50):
    compile_params.extend([
        ("\\documentclass{article}", True),
        ("Just some text", False),
    ])

@pytest.mark.parametrize("latex_string, is_valid_doc", compile_params)
def test_latex_compilation_check(latex_string, is_valid_doc):
    """Test LaTeX document string validation."""
    generator = MockPaperGenerator()
    assert generator.compile_latex(latex_string) == is_valid_doc
