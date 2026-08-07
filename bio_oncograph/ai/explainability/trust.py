"""Trust aggregation module for evaluating model reliability."""

import numpy as np


class TrustAggregator:
    """Aggregates multiple metrics into a single trust score."""

    def __init__(
        self,
        weight_confidence: float = 0.3,
        weight_uncertainty: float = 0.3,
        weight_iqa: float = 0.2,
        weight_calibration: float = 0.2,
    ) -> None:
        """Initializes the TrustAggregator.

        Args:
            weight_confidence: Weight for model confidence.
            weight_uncertainty: Weight for inverse uncertainty.
            weight_iqa: Weight for Image Quality Assessment score.
            weight_calibration: Weight for calibration score.
        """
        total_weight = (
            weight_confidence + weight_uncertainty + weight_iqa + weight_calibration
        )
        self.weight_confidence = weight_confidence / total_weight
        self.weight_uncertainty = weight_uncertainty / total_weight
        self.weight_iqa = weight_iqa / total_weight
        self.weight_calibration = weight_calibration / total_weight

    def compute_trust_score(
        self,
        confidence: float,
        uncertainty: float,
        iqa_score: float,
        calibration_score: float,
    ) -> float:
        """Computes the aggregated scalar trust score.

        Args:
            confidence: Model confidence, typical range [0, 1].
            uncertainty: Model uncertainty. Will be inverted for trust.
            iqa_score: Image quality assessment score, range [0, 1].
            calibration_score: Calibration score (e.g., 1 - ECE), range [0, 1].

        Returns:
            Aggregated trust score in range [0, 1].
        """
        # Ensure inputs are in expected ranges
        confidence_clipped = np.clip(confidence, 0.0, 1.0)
        
        # Invert uncertainty (exponential decay maps [0, inf) to (0, 1])
        inverse_uncertainty = np.exp(-uncertainty)
        
        iqa_clipped = np.clip(iqa_score, 0.0, 1.0)
        calibration_clipped = np.clip(calibration_score, 0.0, 1.0)

        trust_score = (
            self.weight_confidence * confidence_clipped
            + self.weight_uncertainty * inverse_uncertainty
            + self.weight_iqa * iqa_clipped
            + self.weight_calibration * calibration_clipped
        )

        return float(np.clip(trust_score, 0.0, 1.0))
