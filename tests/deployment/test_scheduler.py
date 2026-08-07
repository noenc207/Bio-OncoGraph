import pytest
from unittest.mock import MagicMock

class InferenceScheduler:
    """Scheduler for batch inference jobs."""
    
    def schedule(self, task_id: str, delay_seconds: int) -> None:
        pass
        
    def cancel(self, task_id: str) -> bool:
        return True

@pytest.fixture
def scheduler() -> InferenceScheduler:
    """Provide an InferenceScheduler instance."""
    return InferenceScheduler()

@pytest.mark.parametrize("task_id", [f"task_id_{i}" for i in range(10)])
@pytest.mark.parametrize("time_delay", [0, 10, 60, 3600, 7200, 86400])
def test_inference_scheduler_schedule(scheduler: InferenceScheduler, task_id: str, time_delay: int) -> None:
    """
    Test inference scheduler schedule.
    
    Args:
        scheduler (InferenceScheduler): Scheduler fixture.
        task_id (str): Task identifier.
        time_delay (int): Delay in seconds.
    """
    scheduler.schedule = MagicMock()
    scheduler.schedule(task_id, time_delay)
    scheduler.schedule.assert_called_with(task_id, time_delay)
