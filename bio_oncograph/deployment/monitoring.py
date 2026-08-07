"""Monitoring module for the deployment framework."""
import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MonitoringService:
    """Service to collect and retrieve deployment statistics."""

    def __init__(self) -> None:
        """Initialize the MonitoringService."""
        self._statistics: List[Dict[str, Any]] = []

    def collect_statistics(self, latency_ms: float, memory_mb: float, throughput_rps: float) -> None:
        """Collects runtime statistics.

        Args:
            latency_ms: Request latency in milliseconds.
            memory_mb: Memory usage in megabytes.
            throughput_rps: Throughput in requests per second.
        """
        stat = {
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "memory_mb": memory_mb,
            "throughput_rps": throughput_rps
        }
        self._statistics.append(stat)
        logger.info("Collected statistics: %s", stat)

    def get_statistics(self) -> List[Dict[str, Any]]:
        """Retrieves all collected statistics.

        Returns:
            A list of statistics dictionaries.
        """
        return self._statistics
