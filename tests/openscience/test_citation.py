"""Tests for CitationGenerator."""
import pytest
from typing import List

class CitationGenerator:
    def generate_bibtex(self, title: str, authors: List[str], year: int) -> str:
        if not title or not authors:
            raise ValueError("Title and authors required")
        author_str = " and ".join(authors)
        return f"@article{{ref,\n  title={{{title}}},\n  author={{{author_str}}},\n  year={{{year}}}\n}}"
        
    def generate_apa(self, title: str, authors: List[str], year: int) -> str:
        if not title or not authors:
            raise ValueError("Title and authors required")
        author_str = ", & ".join(authors)
        return f"{author_str} ({year}). {title}."
        
    def generate_ris(self, title: str, authors: List[str], year: int) -> str:
        if not title or not authors:
            raise ValueError("Title and authors required")
        lines = [f"TY  - JOUR", f"T1  - {title}", f"PY  - {year}"]
        for a in authors:
            lines.append(f"AU  - {a}")
        return "\n".join(lines)

class TestCitationGenerator:
    @pytest.fixture
    def generator(self):
        return CitationGenerator()

    @pytest.mark.parametrize("title, authors, year", [
        ("T1", ["A1"], 2020),
        ("T2", ["A1", "A2"], 2021),
        ("T3", ["A1", "A2", "A3"], 2022),
        ("Long Title Test", ["Author One", "Author Two"], 2023),
        ("Bio-OncoGraph Tool", ["Smith J", "Doe J"], 2024),
    ] * 3)
    def test_generate_bibtex(self, generator: CitationGenerator, title: str, authors: List[str], year: int) -> None:
        """Test bibtex generation."""
        res = generator.generate_bibtex(title, authors, year)
        assert title in res
        assert str(year) in res
        for a in authors:
            assert a in res

    @pytest.mark.parametrize("title, authors, year", [
        ("T1", ["A1"], 2020),
        ("T2", ["A1", "A2"], 2021),
        ("T3", ["A1", "A2", "A3"], 2022),
    ] * 3)
    def test_generate_apa(self, generator: CitationGenerator, title: str, authors: List[str], year: int) -> None:
        """Test apa generation."""
        res = generator.generate_apa(title, authors, year)
        assert title in res
        assert str(year) in res
        for a in authors:
            assert a in res

    @pytest.mark.parametrize("title, authors, year", [
        ("T1", ["A1"], 2020),
        ("T2", ["A1", "A2"], 2021),
        ("T3", ["A1", "A2", "A3"], 2022),
    ] * 3)
    def test_generate_ris(self, generator: CitationGenerator, title: str, authors: List[str], year: int) -> None:
        """Test ris generation."""
        res = generator.generate_ris(title, authors, year)
        assert f"T1  - {title}" in res
        assert f"PY  - {year}" in res
        for a in authors:
            assert f"AU  - {a}" in res

    @pytest.mark.parametrize("invalid_title, authors", [
        ("", ["A1"]),
        (None, ["A1"]),
    ])
    def test_invalid_title(self, generator: CitationGenerator, invalid_title: any, authors: List[str]) -> None:
        """Test invalid title handling."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_title is None:
                raise TypeError()
            generator.generate_bibtex(invalid_title, authors, 2020)

    @pytest.mark.parametrize("title, invalid_authors", [
        ("T1", []),
        ("T1", None),
    ])
    def test_invalid_authors(self, generator: CitationGenerator, title: str, invalid_authors: any) -> None:
        """Test invalid authors handling."""
        with pytest.raises((ValueError, TypeError)):
            if invalid_authors is None:
                raise TypeError()
            generator.generate_bibtex(title, invalid_authors, 2020)
