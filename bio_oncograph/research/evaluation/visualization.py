"""Visualization utilities for scientific evaluation."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PlotGenerator:
    """Generator for scientific evaluation plots."""

    @staticmethod
    def plot_bar_chart(data: List[float], title: str) -> str:
        """Generate a bar chart.

        Args:
            data: A list of numerical values.
            title: The title of the plot.

        Returns:
            A mock SVG string representing the plot.
        """
        logger.info("Generating bar chart: %s", title)
        return f"<svg><title>{title} Bar Chart</title><rect/></svg>"

    @staticmethod
    def plot_box_plot(data: List[List[float]], title: str) -> str:
        """Generate a box plot.

        Args:
            data: A list of lists of numerical values, one list per category.
            title: The title of the plot.

        Returns:
            A mock SVG string representing the plot.
        """
        logger.info("Generating box plot: %s", title)
        return f"<svg><title>{title} Box Plot</title><rect/></svg>"

    @staticmethod
    def plot_roc_curve(fpr: List[float], tpr: List[float]) -> str:
        """Generate a Receiver Operating Characteristic (ROC) curve.

        Args:
            fpr: False positive rates.
            tpr: True positive rates.

        Returns:
            A mock SVG string representing the plot.
        """
        logger.info("Generating ROC curve")
        return "<svg><title>ROC Curve</title><path/></svg>"
