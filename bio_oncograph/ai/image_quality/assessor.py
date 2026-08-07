"""
Image Quality Assessor for Bio-OncoGraph.

Provides ``ImageQualityAssessor`` — the top-level entry point
for assessing, scoring, validating, and reporting image quality.

This module orchestrates:
  - **Validators** (``validators.py``)
  - **Metric functions** (``metrics.py``)
  - **Result container** (``report.py``)
  - **Configuration** (``config.py``)

**No deep-learning models are used.**

Design references:
  - Software Architecture § 22 (AI Modules)
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from bio_oncograph.ai.image_quality.config import IQAConfig
from bio_oncograph.ai.image_quality.metrics import (
    compute_artifact_score,
    compute_blur_score,
    compute_brightness_score,
    compute_contrast_score,
    compute_exposure_score,
    compute_focus_score,
    compute_noise_score,
    compute_resolution_score,
)
from bio_oncograph.ai.image_quality.report import ImageQualityResult
from bio_oncograph.ai.image_quality.validators import validate_image

logger = logging.getLogger("BioOncoGraph")


class ImageQualityAssessor:
    """
    Top-level image quality assessor.

    Orchestrates validation, metric computation, and report
    generation.  Fully deterministic and backend-agnostic.

    Example::

        assessor = ImageQualityAssessor()
        result = assessor.assess(image)
        if result.passed:
            print("Image accepted")
        else:
            print(result.warnings)

    Args:
        config: IQA configuration.  Uses defaults when omitted.
    """

    def __init__(
        self, config: IQAConfig = IQAConfig()
    ) -> None:
        """
        Initialise the assessor.

        Args:
            config: Immutable IQA configuration.
        """
        self._config = config

    @property
    def config(self) -> IQAConfig:
        """Return the active configuration."""
        return self._config

    def assess(self, image: np.ndarray) -> ImageQualityResult:
        """
        Perform full image quality assessment.

        Validates the image, computes all quality metrics,
        generates warnings, and returns an immutable result.

        Args:
            image: Input image as ``numpy.ndarray``.

        Returns:
            An ``ImageQualityResult`` with all scores and
            the pass/fail verdict.

        Raises:
            TypeError: If *image* is not an ``ndarray``.
            ValueError: If the image fails basic validation.
        """
        validate_image(image, self._config.minimum_resolution)

        scores = self._compute_all_scores(image)
        overall = self._compute_overall(scores)
        passed = overall >= self._config.minimum_quality_score
        warnings = self._generate_warnings(scores)
        metadata = self._build_metadata(image)

        result = ImageQualityResult(
            overall_score=round(overall, 6),
            passed=passed,
            blur_score=round(scores["blur"], 6),
            focus_score=round(scores["focus"], 6),
            brightness_score=round(scores["brightness"], 6),
            contrast_score=round(scores["contrast"], 6),
            noise_score=round(scores["noise"], 6),
            exposure_score=round(scores["exposure"], 6),
            resolution_score=round(scores["resolution"], 6),
            artifact_score=round(scores["artifact"], 6),
            warnings=tuple(warnings),
            metadata=metadata,
        )

        logger.info(
            "IQA: overall=%.4f passed=%s warnings=%d "
            "shape=%s",
            result.overall_score,
            result.passed,
            len(result.warnings),
            image.shape,
        )
        return result

    def compute_score(self, image: np.ndarray) -> float:
        """
        Compute the overall quality score only.

        This is a lightweight wrapper around ``assess`` that
        discards per-metric details.

        Args:
            image: Input image.

        Returns:
            Overall quality score in ``[0, 1]``.
        """
        return self.assess(image).overall_score

    def validate(self, image: np.ndarray) -> bool:
        """
        Check whether the image passes quality assessment.

        Args:
            image: Input image.

        Returns:
            ``True`` if the image meets the quality threshold.
        """
        return self.assess(image).passed

    def report(self, image: np.ndarray) -> ImageQualityResult:
        """
        Generate a full quality report.

        Alias for ``assess`` — provided for API symmetry.

        Args:
            image: Input image.

        Returns:
            Full ``ImageQualityResult``.
        """
        return self.assess(image)

    # ── Private helpers ─────────────────────────────────────────

    def _compute_all_scores(
        self, image: np.ndarray
    ) -> Dict[str, float]:
        """Compute all eight quality metrics."""
        cfg = self._config
        return {
            "blur": compute_blur_score(
                image, cfg.blur_threshold
            ),
            "focus": compute_focus_score(image),
            "brightness": compute_brightness_score(
                image, cfg.brightness_range
            ),
            "contrast": compute_contrast_score(
                image, cfg.contrast_threshold
            ),
            "noise": compute_noise_score(
                image, cfg.noise_threshold
            ),
            "exposure": compute_exposure_score(image),
            "resolution": compute_resolution_score(
                image, cfg.minimum_resolution
            ),
            "artifact": compute_artifact_score(image),
        }

    def _compute_overall(
        self, scores: Dict[str, float]
    ) -> float:
        """Weighted-average overall score."""
        cfg = self._config
        weights = {
            "blur": cfg.weight_blur,
            "focus": cfg.weight_focus,
            "brightness": cfg.weight_brightness,
            "contrast": cfg.weight_contrast,
            "noise": cfg.weight_noise,
            "exposure": cfg.weight_exposure,
            "resolution": cfg.weight_resolution,
            "artifact": cfg.weight_artifact,
        }
        total_weight = sum(weights.values())
        if total_weight < 1e-12:
            return 0.0
        weighted_sum = sum(
            scores[k] * w for k, w in weights.items()
        )
        return weighted_sum / total_weight

    def _generate_warnings(
        self, scores: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable warnings for low scores."""
        warnings: List[str] = []
        cfg = self._config

        if scores["blur"] < 0.4:
            warnings.append(
                f"Image appears blurry (score={scores['blur']:.3f})."
            )
        if scores["focus"] < 0.4:
            warnings.append(
                f"Image has poor focus (score={scores['focus']:.3f})."
            )
        if scores["brightness"] < 0.5:
            warnings.append(
                f"Brightness out of range "
                f"(score={scores['brightness']:.3f})."
            )
        if scores["contrast"] < 0.4:
            warnings.append(
                f"Low contrast (score={scores['contrast']:.3f})."
            )
        if scores["noise"] < 0.5:
            warnings.append(
                f"High noise level (score={scores['noise']:.3f})."
            )
        if scores["exposure"] < 0.3:
            warnings.append(
                f"Poor exposure (score={scores['exposure']:.3f})."
            )
        if scores["resolution"] < 0.5:
            warnings.append(
                f"Resolution below minimum "
                f"{cfg.minimum_resolution} "
                f"(score={scores['resolution']:.3f})."
            )
        if scores["artifact"] < 0.7:
            warnings.append(
                f"Artifacts detected "
                f"(score={scores['artifact']:.3f})."
            )
        return warnings

    @staticmethod
    def _build_metadata(
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """Build metadata dict from the input image."""
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min_value": float(np.min(image)),
            "max_value": float(np.max(image)),
            "mean_value": float(np.mean(image)),
        }

    def __repr__(self) -> str:
        return (
            f"ImageQualityAssessor("
            f"min_score={self._config.minimum_quality_score})"
        )
