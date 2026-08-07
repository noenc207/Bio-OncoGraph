"""Publication package building and exporting."""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublicationPackage:
    """A complete publication package."""
    title: str
    authors: List[str]
    abstract: str
    figures: Dict[str, np.ndarray] = field(default_factory=dict)
    tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    appendices: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PublicationBuilder:
    """Builder for publication packages."""

    def create_package(self, title: str, authors: List[str], abstract: str) -> PublicationPackage:
        """Create a new publication package.
        
        Args:
            title: Publication title.
            authors: List of authors.
            abstract: Abstract text.
            
        Returns:
            A new PublicationPackage.
        """
        logger.info("Creating publication package: %s", title)
        return PublicationPackage(title=title, authors=authors, abstract=abstract)

    def add_figure(self, package: PublicationPackage, figure_name: str, figure_data: np.ndarray) -> PublicationPackage:
        """Add a figure to the package.
        
        Args:
            package: The current publication package.
            figure_name: Name of the figure.
            figure_data: Numpy array containing figure data/image.
            
        Returns:
            A new PublicationPackage with the added figure.
        """
        new_figures = copy.copy(package.figures)
        new_figures[figure_name] = figure_data
        logger.info("Added figure %s", figure_name)
        return PublicationPackage(
            title=package.title,
            authors=package.authors,
            abstract=package.abstract,
            figures=new_figures,
            tables=package.tables,
            references=package.references,
            appendices=package.appendices,
            metadata=package.metadata
        )

    def add_table(self, package: PublicationPackage, table_name: str, table_data: Dict[str, Any]) -> PublicationPackage:
        """Add a table to the package.
        
        Args:
            package: The current publication package.
            table_name: Name of the table.
            table_data: Table data.
            
        Returns:
            A new PublicationPackage with the added table.
        """
        new_tables = copy.copy(package.tables)
        new_tables[table_name] = table_data
        logger.info("Added table %s", table_name)
        return PublicationPackage(
            title=package.title,
            authors=package.authors,
            abstract=package.abstract,
            figures=package.figures,
            tables=new_tables,
            references=package.references,
            appendices=package.appendices,
            metadata=package.metadata
        )

    def export_latex(self, package: PublicationPackage) -> str:
        """Export the publication package to LaTeX format.
        
        Args:
            package: The publication package.
            
        Returns:
            A LaTeX formatted string.
        """
        author_str = ", ".join(package.authors)
        latex = [
            "\\documentclass{article}",
            f"\\title{{{package.title}}}",
            f"\\author{{{author_str}}}",
            "\\begin{document}",
            "\\maketitle",
            "\\begin{abstract}",
            package.abstract,
            "\\end{abstract}",
            "\\end{document}"
        ]
        return "\n".join(latex)

    def export_markdown(self, package: PublicationPackage) -> str:
        """Export the publication package to Markdown format.
        
        Args:
            package: The publication package.
            
        Returns:
            A Markdown formatted string.
        """
        author_str = ", ".join(package.authors)
        markdown = [
            f"# {package.title}",
            f"**Authors**: {author_str}",
            "",
            "## Abstract",
            package.abstract
        ]
        return "\n".join(markdown)
