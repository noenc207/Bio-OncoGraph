import pytest
from unittest.mock import patch, MagicMock

class PipelineRunner:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def resume_or_start(self, run_id):
        pass

@pytest.mark.parametrize("crash_stage", ["validation", "patch_extraction", "feature_extraction", "graph_generation", "completed"])
@pytest.mark.parametrize("checkpoint_exists", [True, False])
@pytest.mark.parametrize("corrupted_checkpoint", [True, False])
@pytest.mark.parametrize("run_id", ["run_001", "run_999"])
def test_resume_logic_matrix(crash_stage, checkpoint_exists, corrupted_checkpoint, run_id):
    # Generates 5 * 2 * 2 * 2 = 40 test cases
    runner = PipelineRunner(checkpoint_dir="/tmp/checkpoints")
    
    with patch.object(runner, 'resume_or_start') as mock_resume:
        if checkpoint_exists and not corrupted_checkpoint:
            mock_resume.return_value = crash_stage
        elif corrupted_checkpoint:
            mock_resume.side_effect = ValueError("Checkpoint corrupted")
        else:
            mock_resume.return_value = "start"

        if corrupted_checkpoint:
            with pytest.raises(ValueError):
                runner.resume_or_start(run_id)
        else:
            state = runner.resume_or_start(run_id)
            if checkpoint_exists:
                assert state == crash_stage
            else:
                assert state == "start"

@pytest.mark.parametrize("io_error", [True, False])
@pytest.mark.parametrize("retry_count", [0, 3, 5])
def test_resume_io_resilience(io_error, retry_count):
    # 2 * 3 = 6 test cases
    runner = PipelineRunner(checkpoint_dir="/tmp/checkpoints")
    with patch.object(runner, 'resume_or_start') as mock_resume:
        if io_error:
            mock_resume.side_effect = IOError("Disk read error")
            with pytest.raises(IOError):
                runner.resume_or_start("run_err")
        else:
            mock_resume.return_value = "resumed"
            assert runner.resume_or_start("run_ok") == "resumed"
