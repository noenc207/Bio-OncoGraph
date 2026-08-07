"""Table generation utilities for scientific evaluation."""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class TableGenerator:
    """Generator for scientific evaluation tables."""

    @staticmethod
    def generate_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
        """Generate a Markdown formatted table.

        Args:
            headers: List of column headers.
            rows: List of rows, where each row is a list of cell values.

        Returns:
            A string containing the Markdown table.
        """
        logger.info("Generating markdown table")
        if not headers:
            return ""
        header_str = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_strs = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
        return "\n".join([header_str, separator] + row_strs)

    @staticmethod
    def generate_latex_table(headers: List[str], rows: List[List[Any]]) -> str:
        """Generate a LaTeX formatted table.

        Args:
            headers: List of column headers.
            rows: List of rows, where each row is a list of cell values.

        Returns:
            A string containing the LaTeX table.
        """
        logger.info("Generating LaTeX table")
        if not headers:
            return ""
        col_format = "c" * len(headers)
        header_str = " & ".join(headers) + " \\\\"
        row_strs = [" & ".join(str(item) for item in row) + " \\\\" for row in rows]
        
        latex = [
            "\\begin{tabular}{" + col_format + "}",
            header_str,
            "\\hline"
        ]
        latex.extend(row_strs)
        latex.append("\\end{tabular}")
        return "\n".join(latex)

    @staticmethod
    def generate_csv(headers: List[str], rows: List[List[Any]]) -> str:
        """Generate a CSV formatted string.

        Args:
            headers: List of column headers.
            rows: List of rows, where each row is a list of cell values.

        Returns:
            A string containing the CSV data.
        """
        logger.info("Generating CSV table")
        if not headers:
            return ""
        header_str = ",".join(headers)
        row_strs = [",".join(str(item) for item in row) for row in rows]
        return "\n".join([header_str] + row_strs)
