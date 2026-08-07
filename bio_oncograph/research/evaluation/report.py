"""Report definitions and generation for scientific evaluation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationReport:
    """Immutable representation of an evaluation report."""
    report_id: str
    timestamp: float
    benchmark_results: Dict[str, Any]
    ablation_results: Dict[str, Any]
    statistics: Dict[str, Any]
    reproducibility_manifest: Dict[str, Any]


class ReportGenerator:
    """Generator for evaluation reports based on engine state."""

    @staticmethod
    def compile_report(
        report_id: str,
        timestamp: float,
        benchmark: Dict[str, Any],
        ablation: Dict[str, Any],
        stats: Dict[str, Any],
        manifest: Dict[str, Any]
    ) -> EvaluationReport:
        """Compile state into an EvaluationReport.

        Args:
            report_id: Identifier for the report.
            timestamp: Time of report creation.
            benchmark: Benchmark results data.
            ablation: Ablation results data.
            stats: Statistical test results.
            manifest: Reproducibility information.

        Returns:
            A compiled EvaluationReport object.
        """
        logger.info("Compiling EvaluationReport %s", report_id)
        return EvaluationReport(
            report_id=report_id,
            timestamp=timestamp,
            benchmark_results=benchmark,
            ablation_results=ablation,
            statistics=stats,
            reproducibility_manifest=manifest
        )
