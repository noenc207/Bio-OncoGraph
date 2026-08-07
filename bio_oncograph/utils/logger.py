"""
Centralized logging interface for Bio-OncoGraph.

This module provides the ILogger abstract base class and a concrete implementation
that wraps both Python's standard logging and Weights & Biases (wandb).
It adheres strictly to the API Bible contracts for observability.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ILogger(ABC):
    """
    Abstract Interface for experiment loggers.
    """

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """
        Log numerical metrics.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional training step/epoch.
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: DictConfig) -> None:
        """
        Log hyperparameter configuration.

        Args:
            params: Hydra DictConfig object containing all hyperparameters.
        """
        pass

    @abstractmethod
    def log_artifact(self, path: Path, artifact_type: str = "model") -> None:
        """
        Log a file artifact (e.g., model checkpoint, generated heatmap).

        Args:
            path: Path to the artifact file or directory.
            artifact_type: Type of the artifact.
        """
        pass

    @abstractmethod
    def info(self, msg: str) -> None:
        """
        Log an informational message.
        
        Args:
            msg: The message to log.
        """
        pass

    @abstractmethod
    def error(self, msg: str) -> None:
        """
        Log an error message.
        
        Args:
            msg: The error message to log.
        """
        pass


class CentralLogger(ILogger):
    """
    Concrete logger implementation combining standard logging and optional wandb.
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        use_wandb: bool = False,
        log_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the CentralLogger.

        Args:
            project_name: Name of the project (for wandb).
            run_name: Name of the current run.
            use_wandb: Whether to initialize and use wandb.
            log_dir: Optional directory to store local logs.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Setup standard logger
        self._std_logger = logging.getLogger("BioOncoGraph")
        self._std_logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers if initialized multiple times
        if not self._std_logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self._std_logger.addHandler(ch)

            if log_dir is not None:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_path / "experiment.log")
                fh.setFormatter(formatter)
                self._std_logger.addHandler(fh)

        # Setup wandb
        if self.use_wandb:
            wandb.init(project=self.project_name, name=self.run_name)

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """
        Log numerical metrics to std logger and wandb.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional training step/epoch.
        """
        msg = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        prefix = f"[Step {step}] " if step is not None else ""
        self._std_logger.info(f"Metrics: {prefix}{msg}")

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_hyperparams(self, params: DictConfig) -> None:
        """
        Log hyperparameters to std logger and wandb.

        Args:
            params: Hydra DictConfig object containing all hyperparameters.
        """
        yaml_str = OmegaConf.to_yaml(params)
        self._std_logger.info(f"Hyperparameters:\n{yaml_str}")
        
        if self.use_wandb:
            wandb_config = OmegaConf.to_container(params, resolve=True)
            if isinstance(wandb_config, dict):
                wandb.config.update(wandb_config)

    def log_artifact(self, path: Path, artifact_type: str = "model") -> None:
        """
        Log an artifact to wandb and std logger.

        Args:
            path: Path to the artifact file or directory.
            artifact_type: Type of the artifact.
        """
        self._std_logger.info(f"Saved artifact ({artifact_type}) at {path}")
        
        if self.use_wandb:
            artifact = wandb.Artifact(name=path.name, type=artifact_type)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)

    def info(self, msg: str) -> None:
        """
        Log an informational message.
        
        Args:
            msg: The message to log.
        """
        self._std_logger.info(msg)

    def error(self, msg: str) -> None:
        """
        Log an error message.
        
        Args:
            msg: The error message to log.
        """
        self._std_logger.error(msg)
