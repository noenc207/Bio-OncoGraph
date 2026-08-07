"""Preprocessing pipeline configuration."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable configuration for image preprocessing.

    Attributes:
        target_size: Output ``(height, width)``.
        interpolation: Resize method (``nearest``, ``bilinear``, ``bicubic``).
        center_crop: Optional ``(height, width)`` crop after resize.
        padding_mode: Pad mode (``constant``, ``reflect``, ``edge``).
        padding_value: Value for constant padding.
        normalization_mode: ``minmax``, ``meanstd``, or ``none``.
        mean: Per-channel means for mean/std normalisation.
        std: Per-channel stds for mean/std normalisation.
        histogram_equalization: Enable histogram equalisation.
        clahe_enabled: Enable CLAHE.
        clahe_clip_limit: CLAHE clip limit.
        clahe_grid_size: CLAHE tile grid ``(rows, cols)``.
        gamma: Gamma correction exponent (1.0 = no change).
        denoise_enabled: Enable denoising.
        denoise_method: ``gaussian``, ``median``, or ``bilateral``.
        denoise_sigma: Gaussian/bilateral spatial sigma.
        denoise_kernel_size: Kernel size for median filter.
        sharpen_enabled: Enable unsharp-mask sharpening.
        sharpen_strength: Sharpening alpha.
        sharpen_sigma: Gaussian sigma for unsharp mask.
        min_resolution: Minimum ``(h, w)`` for validation.
        max_resolution: Maximum ``(h, w)`` for validation.
    """

    target_size: Tuple[int, int] = (256, 256)
    interpolation: str = "bilinear"
    center_crop: Optional[Tuple[int, int]] = None
    padding_mode: str = "constant"
    padding_value: float = 0.0
    normalization_mode: str = "minmax"
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    histogram_equalization: bool = False
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    gamma: float = 1.0
    denoise_enabled: bool = False
    denoise_method: str = "gaussian"
    denoise_sigma: float = 1.0
    denoise_kernel_size: int = 3
    sharpen_enabled: bool = False
    sharpen_strength: float = 1.0
    sharpen_sigma: float = 1.0
    min_resolution: Tuple[int, int] = (16, 16)
    max_resolution: Tuple[int, int] = (4096, 4096)
