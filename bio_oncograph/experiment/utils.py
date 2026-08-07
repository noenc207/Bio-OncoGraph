"""Utilities for capturing Reproducibility Snapshots."""
import os
import sys
import json
import platform
import subprocess
import logging
from typing import Dict, Any
import psutil

logger = logging.getLogger(__name__)

def capture_reproducibility_snapshot(output_dir: str) -> None:
    """Captures system state for exact reproducibility.
    
    Args:
        output_dir: Directory to save the environment.json and reproducibility.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    env_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    
    # Capture GPU info if available
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip().split("\n")
        env_info["gpus"] = gpu_info
    except (subprocess.SubprocessError, FileNotFoundError):
        env_info["gpus"] = []
        
    # Capture git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        git_commit = "unknown"
        
    # Capture pip freeze
    try:
        pip_freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).strip().split("\n")
    except subprocess.SubprocessError:
        pip_freeze = []
        
    reproducibility_info = {
        "git_commit": git_commit,
        "dependencies": pip_freeze
    }
    
    with open(os.path.join(output_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=4)
        
    with open(os.path.join(output_dir, "reproducibility.json"), "w", encoding="utf-8") as f:
        json.dump(reproducibility_info, f, indent=4)
        
    logger.info(f"Reproducibility snapshot saved to {output_dir}")
