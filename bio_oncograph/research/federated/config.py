"""Federated learning configuration module."""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FederatedConfig:
    """Configuration for federated learning.

    Attributes:
        aggregation_method: The aggregation method to use ('fedavg', 'fedprox', 'fedopt', 'robust').
        privacy_method: The privacy preserving method to use ('none', 'dp', 'secure_agg').
        communication_mode: The communication mode ('sync', 'async').
        total_rounds: The total number of federated training rounds.
        clients_per_round: The number of clients selected per round.
        straggler_timeout: The timeout for straggling clients in seconds.
        dp_epsilon: Differential privacy epsilon parameter.
        dp_delta: Differential privacy delta parameter.
        seed: Random seed for reproducibility.
    """
    aggregation_method: str = "fedavg"
    privacy_method: str = "none"
    communication_mode: str = "sync"
    total_rounds: int = 10
    clients_per_round: int = 3
    straggler_timeout: float = 60.0
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    seed: int = 42
