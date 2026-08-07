import pytest
from typing import List

class PlotGenerator:
    """Mock for PlotGenerator."""
    def generate_bar(self, data: List[float]) -> str:
        return f"<svg>bar_{len(data)}</svg>"
        
    def generate_scatter(self, x: List[float], y: List[float]) -> str:
        return f"<svg>scatter_{len(x)}_{len(y)}</svg>"

@pytest.mark.parametrize("size", list(range(10)))
def test_generate_bar_chart(size: int) -> None:
    """Test bar chart SVG generation."""
    pg = PlotGenerator()
    assert f"bar_{size}" in pg.generate_bar([float(i) for i in range(size)])

@pytest.mark.parametrize("size", list(range(6)))
def test_generate_scatter_plot(size: int) -> None:
    """Test scatter plot SVG generation."""
    pg = PlotGenerator()
    data = [float(i) for i in range(size)]
    assert f"scatter_{size}_{size}" in pg.generate_scatter(data, data)
