"""Distributed tracing module for tracking requests through the platform."""
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DistributedTracer:
    """Tracks execution lineage and request flows."""

    def __init__(self) -> None:
        """Initializes the DistributedTracer."""
        self._active_traces: Dict[str, float] = {}
        self._completed_traces: Dict[str, Dict[str, Any]] = {}

    def start_trace(self, trace_id: str) -> None:
        """Starts tracking a new trace.

        Args:
            trace_id: Unique identifier for the trace.
        """
        self._active_traces[trace_id] = time.time()
        logger.info("Started trace %s", trace_id)

    def end_trace(self, trace_id: str, status: str = "success") -> None:
        """Ends an active trace.

        Args:
            trace_id: Unique identifier for the trace.
            status: Final status of the trace (e.g., 'success', 'error').
        """
        if trace_id in self._active_traces:
            start_time = self._active_traces.pop(trace_id)
            duration = time.time() - start_time
            self._completed_traces[trace_id] = {
                "duration_sec": duration,
                "status": status
            }
            logger.info("Ended trace %s with status %s", trace_id, status)
        else:
            logger.warning("Attempted to end unknown trace %s", trace_id)

    def extract_lineage(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Extracts lineage information for a completed trace.

        Args:
            trace_id: Unique identifier for the trace.

        Returns:
            A dictionary with lineage details if the trace exists, otherwise None.
        """
        return self._completed_traces.get(trace_id)
