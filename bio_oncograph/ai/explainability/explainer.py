"""Explainability engine module."""

from typing import Any, Dict, Optional
import numpy as np

from bio_oncograph.ai.explainability.config import ExplainabilityConfig
from bio_oncograph.ai.explainability.outputs import (
    ExplainabilityOutput,
    TrustScoreOutput,
    UncertaintyOutput,
)
from bio_oncograph.ai.explainability.report import ExplainabilityReport
from bio_oncograph.ai.explainability.validation import validate_explainability_inputs


class ExplainabilityEngine:
    """Core engine for explainability, uncertainty, and trust scoring."""

    def __init__(self, config: ExplainabilityConfig) -> None:
        """Initializes the ExplainabilityEngine.

        Args:
            config: Configuration for the explainability framework.
        """
        self.config = config

    def explain(
        self, model_output: Dict[str, Any], graph: Optional[Any] = None
    ) -> ExplainabilityOutput:
        """Computes explainability attributions.

        Args:
            model_output: Dictionary containing model outputs.
            graph: Optional graph representation.

        Returns:
            ExplainabilityOutput containing feature attributions.
        """
        logits = model_output['logits']
        # Mock attribution computation
        attributions = np.abs(logits) / (np.sum(np.abs(logits), axis=-1, keepdims=True) + 1e-8)

        return ExplainabilityOutput(
            scores=attributions,
            metadata={"method": self.config.attribution_method},
            warnings=[],
            version="1.0.0",
        )

    def estimate_uncertainty(self, model_output: Dict[str, Any]) -> UncertaintyOutput:
        """Estimates prediction uncertainty.

        Args:
            model_output: Dictionary containing model outputs.

        Returns:
            UncertaintyOutput containing uncertainty scores.
        """
        logits = model_output['logits']
        # Compute entropy as a measure of uncertainty
        logits = np.clip(logits, -50.0, 50.0)  # Avoid overflow
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)

        return UncertaintyOutput(
            scores=entropy,
            metadata={"method": "entropy"},
            warnings=[],
            version="1.0.0",
        )

    def compute_trust_score(
        self,
        model_output: Dict[str, Any],
        uncertainty: UncertaintyOutput,
        iqa_score: float,
    ) -> TrustScoreOutput:
        """Computes a trust score for the prediction.

        Args:
            model_output: Dictionary containing model outputs.
            uncertainty: UncertaintyOutput from estimate_uncertainty.
            iqa_score: Image quality assessment score (or similar modality quality).

        Returns:
            TrustScoreOutput containing trust scores.
        """
        # A simple mock trust score formula
        mean_uncertainty = float(np.mean(uncertainty.scores))
        
        # High uncertainty -> lower trust, low IQA -> lower trust
        aggregated_trust = max(0.0, min(1.0, iqa_score - (0.5 * mean_uncertainty)))
        trust_scores = np.ones_like(uncertainty.scores) * aggregated_trust

        return TrustScoreOutput(
            scores=trust_scores,
            aggregated_trust_score=aggregated_trust,
            metadata={"iqa_score": iqa_score},
            warnings=[],
            version="1.0.0",
        )

    def validate(
        self, model_output: Dict[str, Any], graph: Optional[Any] = None
    ) -> bool:
        """Validates the input using the validation module.

        Args:
            model_output: Dictionary containing model outputs.
            graph: Optional graph representation.

        Returns:
            True if valid, False otherwise.
        """
        try:
            validate_explainability_inputs(model_output, graph)
            return True
        except ValueError:
            return False

    def report(
        self,
        model_output: Dict[str, Any],
        graph: Optional[Any] = None,
        iqa_score: float = 1.0,
    ) -> ExplainabilityReport:
        """Generates a full explainability report.

        Args:
            model_output: Dictionary containing model outputs.
            graph: Optional graph representation.
            iqa_score: Quality score of the input.

        Returns:
            ExplainabilityReport consolidating all computed metrics.
        """
        is_valid = self.validate(model_output, graph)
        if not is_valid:
            return ExplainabilityReport(
                is_valid=False,
                explainability=None,
                uncertainty=None,
                trust=None,
                metadata={"error": "Input validation failed."},
            )

        explain_out = None
        if self.config.enable_explainability:
            explain_out = self.explain(model_output, graph)

        uncertainty_out = None
        if self.config.enable_uncertainty:
            uncertainty_out = self.estimate_uncertainty(model_output)

        trust_out = None
        if self.config.enable_trust_score and uncertainty_out is not None:
            trust_out = self.compute_trust_score(
                model_output, uncertainty_out, iqa_score
            )

        return ExplainabilityReport(
            is_valid=True,
            explainability=explain_out,
            uncertainty=uncertainty_out,
            trust=trust_out,
            metadata={"version": "1.0.0"},
        )
