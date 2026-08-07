"""
IQA configuration for Bio-OncoGraph.

Provides ``IQAConfig`` — an immutable configuration dataclass
holding all thresholds and parameters for image quality assessment.

Design references:
  - Software Architecture § 22 (AI Modules)
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class IQAConfig:
    """
    Configuration for Image Quality Assessment.

    All thresholds define the boundary between acceptable and
    unacceptable quality.  Scores are normalised to ``[0, 1]``
    where ``1.0`` is best.

    Attributes:
        minimum_quality_score: Overall score below which the
            image is rejected.
        minimum_resolution: ``(height, width)`` minimum in pixels.
        blur_threshold: Variance-of-Laplacian value that maps to
            a blur score of ``1.0``.  Lower raw values (smoother
            images) produce lower scores.
        brightness_range: ``(low, high)`` normalised mean-intensity
            range considered acceptable.
        contrast_threshold: Normalised standard-deviation value
            that maps to a contrast score of ``1.0``.
        noise_threshold: Estimated noise sigma above which the
            noise score drops to ``0.0``.
        weight_blur: Weight for blur score in overall computation.
        weight_focus: Weight for focus score.
        weight_brightness: Weight for brightness score.
        weight_contrast: Weight for contrast score.
        weight_noise: Weight for noise score.
        weight_exposure: Weight for exposure score.
        weight_resolution: Weight for resolution score.
        weight_artifact: Weight for artifact score.
    """

    minimum_quality_score: float = 0.5
    minimum_resolution: Tuple[int, int] = (32, 32)
    blur_threshold: float = 100.0
    brightness_range: Tuple[float, float] = (0.1, 0.9)
    contrast_threshold: float = 0.1
    noise_threshold: float = 50.0
    weight_blur: float = 0.15
    weight_focus: float = 0.15
    weight_brightness: float = 0.10
    weight_contrast: float = 0.10
    weight_noise: float = 0.15
    weight_exposure: float = 0.10
    weight_resolution: float = 0.15
    weight_artifact: float = 0.10
