"""Artifact management and exporting for scientific evaluation."""

import json
import logging
import os
from typing import Any, Dict

from .report import EvaluationReport

logger = logging.getLogger(__name__)


class ArtifactExporter:
    """Exporter for evaluation artifacts."""

    @staticmethod
    def export_json(data: Dict[str, Any], path: str) -> None:
        """Export data to a JSON file.

        Args:
            data: Dictionary to be exported.
            path: Target file path.
        """
        logger.info("Exporting JSON to %s", path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_text(text: str, path: str) -> None:
        """Export text to a file.

        Args:
            text: Text content to be exported.
            path: Target file path.
        """
        logger.info("Exporting text to %s", path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    @staticmethod
    def export_all(report: EvaluationReport, out_dir: str) -> None:
        """Export all components of an evaluation report to a directory.

        Args:
            report: The evaluation report to export.
            out_dir: The target output directory.
        """
        logger.info("Exporting all artifacts to %s", out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        ArtifactExporter.export_json(
            {"report_id": report.report_id, "timestamp": report.timestamp},
            os.path.join(out_dir, "metadata.json")
        )
        ArtifactExporter.export_json(report.benchmark_results, os.path.join(out_dir, "benchmark.json"))
        ArtifactExporter.export_json(report.ablation_results, os.path.join(out_dir, "ablation.json"))
        ArtifactExporter.export_json(report.statistics, os.path.join(out_dir, "statistics.json"))
        ArtifactExporter.export_json(report.reproducibility_manifest, os.path.join(out_dir, "manifest.json"))
