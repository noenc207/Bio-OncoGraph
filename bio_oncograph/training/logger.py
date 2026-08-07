"""Training logger module for tracking metrics and environment information."""

import csv
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """Handles logging of training metrics and environment information.

    Supports TensorBoard, CSV, JSON, and Console logging.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        use_csv: bool = True,
        use_json: bool = True,
        console_level: int = logging.INFO,
    ) -> None:
        """Initializes the TrainingLogger.

        Args:
            log_dir: Directory to save log files.
            use_tensorboard: Whether to use TensorBoard logging.
            use_csv: Whether to save metrics to a CSV file.
            use_json: Whether to save metrics to a JSON file.
            console_level: Logging level for console output.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        self.use_json = use_json

        # Set up console logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(console_level)
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(console_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Set up TensorBoard
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir)) if use_tensorboard else None

        # Set up CSV
        self.csv_path = self.log_dir / "metrics.csv" if use_csv else None
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers_written = False

        # Set up JSON
        self.json_path = self.log_dir / "metrics.json" if use_json else None
        self.json_data: list[Dict[str, Any]] = []

    def log_environment(self) -> None:
        """Logs environment information such as Python version, PyTorch version, etc."""
        env_info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
        }

        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["cudnn_version"] = torch.backends.cudnn.version()
            env_info["gpu_count"] = torch.cuda.device_count()
            env_info["gpu_name"] = torch.cuda.get_device_name(0)

        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )
            env_info["git_commit"] = git_commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            env_info["git_commit"] = "Unknown"

        self.logger.info("Environment Info:")
        for key, value in env_info.items():
            self.logger.info(f"  {key}: {value}")

        env_file = self.log_dir / "environment.json"
        with open(env_file, "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=4)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Logs metrics to the configured destinations.

        Args:
            metrics: Dictionary of metric names and values.
            step: The current training step or epoch.
        """
        # Console
        metrics_str = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        self.logger.info(f"Step {step} | {metrics_str}")

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)

        # CSV
        if self.csv_path is not None:
            if self.csv_file is None:
                self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            
            row = {"step": step, **metrics}
            if not self.csv_headers_written:
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=["step"] + list(metrics.keys()))
                self.csv_writer.writeheader()
                self.csv_headers_written = True
            
            if self.csv_writer is not None:
                self.csv_writer.writerow(row)
                self.csv_file.flush()

        # JSON
        if self.json_path is not None:
            self.json_data.append({"step": step, **metrics})
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.json_data, f, indent=4)

    def close(self) -> None:
        """Closes all open resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()
