"""
Unit tests for the centralized logger.
Ensures compliance with API contracts and graceful degradation.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from bio_oncograph.utils.logger import CentralLogger


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for logs."""
    return tmp_path


def test_central_logger_initialization_no_wandb(temp_log_dir: Path) -> None:
    """Test logger initialization without wandb falls back to std logging cleanly."""
    logger = CentralLogger(
        project_name="test_proj",
        run_name="test_run",
        use_wandb=False,
        log_dir=temp_log_dir
    )
    
    assert logger.use_wandb is False
    assert (temp_log_dir / "experiment.log").exists()


@patch("bio_oncograph.utils.logger.wandb")
def test_central_logger_initialization_with_wandb(mock_wandb: MagicMock, temp_log_dir: Path) -> None:
    """Test logger initialization correctly calls wandb.init."""
    with patch("bio_oncograph.utils.logger.WANDB_AVAILABLE", True):
        logger = CentralLogger(
            project_name="test_proj",
            run_name="test_run",
            use_wandb=True,
            log_dir=temp_log_dir
        )
        
        assert logger.use_wandb is True
        mock_wandb.init.assert_called_once_with(project="test_proj", name="test_run")


def test_log_metrics_std_only(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_metrics writes correct format to standard logger."""
    logger = CentralLogger(project_name="p", run_name="r", use_wandb=False)
    
    with caplog.at_level(logging.INFO):
        logger.log_metrics({"c_index": 0.85, "loss": 0.1}, step=10)
    
    assert "Metrics: [Step 10] c_index: 0.8500 | loss: 0.1000" in caplog.text


@patch("bio_oncograph.utils.logger.wandb")
def test_log_metrics_with_wandb(mock_wandb: MagicMock) -> None:
    """Test log_metrics delegates to wandb.log."""
    with patch("bio_oncograph.utils.logger.WANDB_AVAILABLE", True):
        logger = CentralLogger(project_name="p", run_name="r", use_wandb=True)
        metrics = {"loss": 0.5}
        logger.log_metrics(metrics, step=5)
        
        mock_wandb.log.assert_called_once_with(metrics, step=5)


def test_log_hyperparams_std(caplog: pytest.LogCaptureFixture) -> None:
    """Test log_hyperparams serializes OmegaConf DictConfig."""
    logger = CentralLogger(project_name="p", run_name="r", use_wandb=False)
    cfg = OmegaConf.create({"model": {"hidden_dim": 256}})
    
    with caplog.at_level(logging.INFO):
        logger.log_hyperparams(cfg)
        
    assert "hidden_dim: 256" in caplog.text


def test_info_and_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test standard info and error logging routes."""
    logger = CentralLogger(project_name="p", run_name="r", use_wandb=False)
    
    with caplog.at_level(logging.INFO):
        logger.info("Info message")
        logger.error("Error message")
        
    assert "Info message" in caplog.text
    assert "Error message" in caplog.text
