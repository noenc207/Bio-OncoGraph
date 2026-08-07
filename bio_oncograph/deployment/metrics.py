"""Metrics collection for platform hardware and performance monitoring."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and aggregates hardware and performance metrics."""

    def __init__(self) -> None:
        """Initializes the MetricsCollector."""
        self._metrics: List[Dict[str, float]] = []

    def aggregate(self, latency_ms: float, throughput_rps: float, cpu_utilization: float, gpu_utilization: float) -> None:
        """Aggregates system and performance metrics.

        Args:
            latency_ms: Request latency in milliseconds.
            throughput_rps: Requests per second.
            cpu_utilization: CPU usage percentage (0-100).
            gpu_utilization: GPU usage percentage (0-100).
        """
        metric_record = {
            "latency_ms": latency_ms,
            "throughput_rps": throughput_rps,
            "cpu_utilization": cpu_utilization,
            "gpu_utilization": gpu_utilization
        }
        self._metrics.append(metric_record)
        logger.info("Aggregated metrics: CPU %.1f%%, GPU %.1f%%", cpu_utilization, gpu_utilization)
