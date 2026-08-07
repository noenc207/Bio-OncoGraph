import logging
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeploymentConfig:
    """Configuration for the deployment framework.

    Attributes:
        host: Host address for the deployment service.
        port: Port for the deployment service.
        mode: Deployment mode, e.g., 'local' or 'batch'.
        monitoring_enabled: Flag indicating if monitoring is enabled.
        drift_detection_enabled: Flag indicating if data drift detection is enabled.
        cache_dir: Directory path for caching models and artifacts.
        seed: Random seed for reproducibility.
    """
    host: str
    port: int
    mode: str
    monitoring_enabled: bool
    drift_detection_enabled: bool
    cache_dir: str
    seed: Optional[int] = None
