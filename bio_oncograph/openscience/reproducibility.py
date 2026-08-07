"""Reproducibility tracking and environment capture tools."""

import logging
import platform
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ReproducibilityPackage:
    """A complete package for reproducing an experiment."""
    config_snapshot: Dict[str, Any]
    software_versions: Dict[str, str]
    dependency_manifest: Dict[str, str]
    experiment_metadata: Dict[str, Any]
    random_seeds: List[int]
    hardware_metadata: Dict[str, str]
    timestamp: float

class ReproducibilityExporter:
    """Handles capturing environments and exporting reproducibility packages."""
    
    def capture_environment(self) -> Dict[str, str]:
        """Captures hardware and software environment details.
        
        Returns:
            Dict[str, str]: A dictionary of environment details.
        """
        logger.info("Capturing runtime environment.")
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy_version": np.__version__
        }
        
    def export_package(self, config: Dict[str, Any], seeds: List[int]) -> ReproducibilityPackage:
        """Exports a full reproducibility package.
        
        Args:
            config (Dict[str, Any]): The configuration snapshot.
            seeds (List[int]): Random seeds used.
            
        Returns:
            ReproducibilityPackage: The compiled package.
        """
        logger.info("Exporting reproducibility package.")
        env_details = self.capture_environment()
        
        return ReproducibilityPackage(
            config_snapshot=config,
            software_versions={"python": env_details.get("python_version", "unknown")},
            dependency_manifest={"numpy": env_details.get("numpy_version", "unknown")},
            experiment_metadata={"status": "exported"},
            random_seeds=seeds,
            hardware_metadata=env_details,
            timestamp=datetime.datetime.now().timestamp()
        )
        
    def verify_package(self, package: ReproducibilityPackage) -> bool:
        """Verifies the integrity and completeness of a reproducibility package.
        
        Args:
            package (ReproducibilityPackage): The package to verify.
            
        Returns:
            bool: True if the package is valid, False otherwise.
        """
        logger.info("Verifying reproducibility package from timestamp %s", package.timestamp)
        if not package.config_snapshot or not package.random_seeds:
            logger.warning("Package missing critical config or seeds.")
            return False
        return True
