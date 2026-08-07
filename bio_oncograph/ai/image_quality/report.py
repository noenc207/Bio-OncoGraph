"""
Image Quality Assessment result container.

Provides ``ImageQualityResult`` — an immutable, frozen dataclass
holding all per-metric scores, the overall quality verdict, any
warnings, and metadata.

Design references:
  - Software Architecture § 22 (AI Modules)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ImageQualityResult:
    """
    Immutable result of an image quality assessment.

    All ``*_score`` fields are normalised to ``[0, 1]`` where
    ``1.0`` is best quality.

    Attributes:
        overall_score: Weighted aggregate quality score.
        passed: ``True`` if the image meets the minimum quality
            threshold.
        blur_score: Sharpness via Variance of Laplacian.
        focus_score: Focus quality via gradient magnitude.
        brightness_score: Brightness appropriateness.
        contrast_score: Image contrast.
        noise_score: Inverse noise level (higher = less noisy).
        exposure_score: Histogram spread.
        resolution_score: Resolution adequacy.
        artifact_score: Absence of artifacts (higher = fewer).
        warnings: Human-readable quality warnings.
        metadata: Auxiliary information (image shape, dtype, etc.).
    """

    overall_score: float
    passed: bool
    blur_score: float
    focus_score: float
    brightness_score: float
    contrast_score: float
    noise_score: float
    exposure_score: float
    resolution_score: float
    artifact_score: float
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by overall_score and passed flag."""
        return hash((self.overall_score, self.passed))
