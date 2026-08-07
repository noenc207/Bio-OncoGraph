import logging
import uuid
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class InferenceScheduler:
    """Schedules and manages batch inference tasks."""

    def __init__(self) -> None:
        """Initializes the InferenceScheduler."""
        self._scheduled_tasks: Dict[str, Dict[str, str]] = {}

    def schedule_batch(
        self, deployment_id: str, input_dir: str, output_dir: str, schedule_expr: str
    ) -> str:
        """Schedules a batch inference job.

        Args:
            deployment_id: The identifier of the deployment to execute the batch job.
            input_dir: Directory containing the input data.
            output_dir: Directory to save the inference results.
            schedule_expr: Cron expression defining the schedule (e.g., '0 0 * * *').

        Returns:
            The unique identifier (job_id) for the scheduled batch job.
        """
        job_id = str(uuid.uuid4())
        
        self._scheduled_tasks[job_id] = {
            "deployment_id": deployment_id,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "schedule_expr": schedule_expr,
            "created_at": datetime.utcnow().isoformat(),
            "status": "SCHEDULED",
        }
        
        logger.info(
            f"Scheduled batch job {job_id} for deployment {deployment_id} "
            f"with schedule '{schedule_expr}'."
        )
        return job_id

    def cancel_batch(self, job_id: str) -> bool:
        """Cancels a scheduled batch job.

        Args:
            job_id: The identifier of the batch job to cancel.

        Returns:
            True if successfully canceled, False if the job was not found.
        """
        if job_id in self._scheduled_tasks:
            self._scheduled_tasks[job_id]["status"] = "CANCELED"
            logger.info(f"Canceled batch job {job_id}.")
            return True
        
        logger.warning(f"Failed to cancel batch job {job_id}: not found.")
        return False
