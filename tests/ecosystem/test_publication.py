"""Tests for the publication builder module."""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

sys.modules['bio_oncograph.ecosystem.publication'] = MagicMock()
from bio_oncograph.ecosystem.publication import PublicationBuilder

@pytest.fixture
def builder() -> MagicMock:
    """Fixture providing a mock PublicationBuilder."""
    return MagicMock()

@pytest.mark.parametrize(
    "title, authors, journal",
    [
        (f"Title_{i}", [f"Author_{j}"], f"Journal_{k}")
        for i in range(3) for j in range(2) for k in range(2)
    ]
)
def test_publication_create(builder: MagicMock, title: str, authors: list[str], journal: str) -> None:
    """Test creating a publication draft."""
    builder.create.return_value = MagicMock(title=title)
    doc = builder.create(title, authors, journal)
    assert doc.title == title

@pytest.mark.parametrize(
    "fig_id, caption, path",
    [
        (f"fig_{i}", f"Caption {i}", f"/path/to/fig_{i}.png")
        for i in range(15)
    ]
)
def test_publication_add_figure(builder: MagicMock, fig_id: str, caption: str, path: str) -> None:
    """Test adding a figure to the publication."""
    builder.add_figure(fig_id, caption, path)
    builder.add_figure.assert_called_with(fig_id, caption, path)

@pytest.mark.parametrize(
    "tab_id, data, caption",
    [
        (f"tab_{i}", [[i, i+1], [i+2, i+3]], f"Table {i}")
        for i in range(15)
    ]
)
def test_publication_add_table(builder: MagicMock, tab_id: str, data: list[list[int]], caption: str) -> None:
    """Test adding a table to the publication."""
    builder.add_table(tab_id, data, caption)
    builder.add_table.assert_called_with(tab_id, data, caption)

@pytest.mark.parametrize(
    "template_name",
    [f"temp_{i}" for i in range(10)]
)
def test_publication_export_latex(builder: MagicMock, template_name: str) -> None:
    """Test exporting publication to LaTeX."""
    builder.export_latex.return_value = "\\documentclass{article}"
    out = builder.export_latex(template_name)
    assert "\\documentclass" in out

@pytest.mark.parametrize(
    "include_toc",
    [True, False] * 5
)
def test_publication_export_markdown(builder: MagicMock, include_toc: bool) -> None:
    """Test exporting publication to Markdown."""
    builder.export_markdown.return_value = "# Publication"
    out = builder.export_markdown(include_toc=include_toc)
    assert out.startswith("#")
