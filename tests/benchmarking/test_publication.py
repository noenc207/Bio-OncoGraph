"""Tests for publication-ready outputs like TeX tables."""

import pytest

def generate_tex_table(model, m1, m2, bold):
    """Mock tex table generator."""
    row = f"{model} & {m1:.3f} & {m2:.3f} \\\\"
    if bold:
        row = f"\\textbf{{{model}}} & \\textbf{{{m1:.3f}}} & \\textbf{{{m2:.3f}}} \\\\"
    return row

@pytest.mark.parametrize("model_name", ["Model A", "Model B", "Model_C^2", "Baseline", "Proposed"])
@pytest.mark.parametrize("metric1", [0.95, 0.88, 0.12, 1.00, 0.00])
@pytest.mark.parametrize("metric2", [0.01, 0.05, 0.10, 0.50, 0.99])
@pytest.mark.parametrize("bold_highest", [True, False])
def test_tex_table_generation(model_name, metric1, metric2, bold_highest):
    """Verify TeX table generation strings are correctly formatted."""
    tex_str = generate_tex_table(model_name, metric1, metric2, bold_highest)
    
    assert model_name in tex_str
    if bold_highest:
        assert "\\textbf" in tex_str
    assert "\\\\" in tex_str
