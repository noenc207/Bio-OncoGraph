"""Utilities for generating academic citations."""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CitationMetadata:
    """Metadata required to generate a citation."""
    title: str
    authors: List[str]
    year: int
    journal: str
    doi: str

class CitationGenerator:
    """Generator for various citation formats."""
    
    def generate_bibtex(self, title: str, authors: List[str], year: int, journal: str, doi: str) -> str:
        """Generates a BibTeX citation.
        
        Args:
            title (str): Title of the work.
            authors (List[str]): List of authors.
            year (int): Year of publication.
            journal (str): Journal name.
            doi (str): Digital Object Identifier.
            
        Returns:
            str: Formatted BibTeX string.
        """
        logger.info("Generating BibTeX citation for %s", doi)
        first_author_last_name = authors[0].split()[-1] if authors else "Unknown"
        bib_id = f"{first_author_last_name}{year}"
        
        authors_str = " and ".join(authors)
        return f"""@article{{{bib_id},
  title={{{title}}},
  author={{{authors_str}}},
  journal={{{journal}}},
  year={{{year}}},
  doi={{{doi}}}
}}"""

    def generate_apa(self, title: str, authors: List[str], year: int, journal: str, doi: str) -> str:
        """Generates an APA-style citation.
        
        Args:
            title (str): Title of the work.
            authors (List[str]): List of authors.
            year (int): Year of publication.
            journal (str): Journal name.
            doi (str): Digital Object Identifier.
            
        Returns:
            str: Formatted APA citation string.
        """
        logger.info("Generating APA citation for %s", doi)
        authors_str = ", ".join(authors)
        return f"{authors_str} ({year}). {title}. {journal}. https://doi.org/{doi}"

    def generate_ris(self, title: str, authors: List[str], year: int, journal: str, doi: str) -> str:
        """Generates a RIS format citation.
        
        Args:
            title (str): Title of the work.
            authors (List[str]): List of authors.
            year (int): Year of publication.
            journal (str): Journal name.
            doi (str): Digital Object Identifier.
            
        Returns:
            str: Formatted RIS string.
        """
        logger.info("Generating RIS citation for %s", doi)
        lines = ["TY  - JOUR", f"TI  - {title}", f"PY  - {year}", f"JO  - {journal}", f"DO  - {doi}"]
        for author in authors:
            lines.append(f"AU  - {author}")
        lines.append("ER  - ")
        return "\n".join(lines)
