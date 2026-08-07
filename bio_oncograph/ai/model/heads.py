"""Prediction Heads for Bio-OncoGraph."""
import numpy as np
from typing import Dict, Any


class ClassificationHead:
    """Classification head for discrete outcomes.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        weights (np.ndarray): Linear layer weights.
        bias (np.ndarray): Linear layer bias.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ClassificationHead.

        Args:
            config: Configuration containing 'input_dim', 'num_classes', and 'seed'.
        """
        self.config = config
        seed = config.get("seed", 42)
        np.random.seed(seed)
        
        input_dim = config.get("input_dim", 64)
        num_classes = config.get("num_classes", 2)
        
        self.weights = np.random.randn(input_dim, num_classes) * np.sqrt(2.0 / (input_dim + num_classes))
        self.bias = np.zeros(num_classes)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """Compute logits from features.

        Args:
            features: Fused features array.

        Returns:
            Logits array.
        """
        logits = features.dot(self.weights) + self.bias
        return logits


class SurvivalHead:
    """Survival head for continuous risk scoring.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        weights (np.ndarray): Linear layer weights.
        bias (np.ndarray): Linear layer bias.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SurvivalHead.

        Args:
            config: Configuration containing 'input_dim' and 'seed'.
        """
        self.config = config
        seed = config.get("seed", 42)
        np.random.seed(seed)
        
        input_dim = config.get("input_dim", 64)
        
        self.weights = np.random.randn(input_dim, 1) * np.sqrt(2.0 / (input_dim + 1))
        self.bias = np.zeros(1)

    def forward(self, features: np.ndarray) -> float:
        """Compute risk score from features.

        Args:
            features: Fused features array.

        Returns:
            A risk score (float).
        """
        risk_score = features.dot(self.weights) + self.bias
        return float(risk_score[0]) if risk_score.size == 1 else float(risk_score)
