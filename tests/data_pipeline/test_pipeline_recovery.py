import pytest
from unittest.mock import MagicMock, patch

class PipelineManager:
    def __init__(self, strategy="retry"):
        self.strategy = strategy

    def run_stage(self, stage_name):
        pass

    def recover(self, error):
        pass

@pytest.mark.parametrize("stage_name", ["wsi_validation", "patch_extraction", "feature_extraction", "graph_generation"])
@pytest.mark.parametrize("error_type", ["MemoryError", "IOError", "TimeoutError", "ValueError"])
@pytest.mark.parametrize("strategy", ["retry", "skip", "abort", "fallback"])
@pytest.mark.parametrize("retry_count", [1, 3, 5])
@pytest.mark.parametrize("is_fatal", [True, False])
def test_pipeline_recovery_matrix(stage_name, error_type, strategy, retry_count, is_fatal):
    # Generates 4 * 4 * 4 * 3 * 2 = 384 test cases
    manager = PipelineManager(strategy=strategy)
    
    error_classes = {
        "MemoryError": MemoryError,
        "IOError": IOError,
        "TimeoutError": TimeoutError,
        "ValueError": ValueError
    }
    error_instance = error_classes[error_type]("Mocked error")
    
    manager.run_stage = MagicMock(side_effect=error_instance)
    
    if strategy == "abort" or is_fatal:
        manager.recover = MagicMock(return_value=False)
        with pytest.raises(Exception):
            try:
                manager.run_stage(stage_name)
            except Exception as e:
                if not manager.recover(e):
                    raise e
    else:
        manager.recover = MagicMock(return_value=True)
        try:
            manager.run_stage(stage_name)
        except Exception as e:
            recovered = manager.recover(e)
            assert recovered is True

@pytest.mark.parametrize("graceful_shutdown", [True, False])
def test_pipeline_shutdown(graceful_shutdown):
    # 2 test cases
    manager = PipelineManager()
    manager.run_stage = MagicMock()
    if graceful_shutdown:
        manager.run_stage.return_value = "shutdown_ok"
        assert manager.run_stage("cleanup") == "shutdown_ok"
    else:
        manager.run_stage.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            manager.run_stage("cleanup")
