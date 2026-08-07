"""
Image Preprocessing Pipeline for Bio-OncoGraph.

Provides ``ImagePreprocessingPipeline`` — the top-level entry point
that builds and executes a configurable, deterministic sequence of
classical image transforms.

**No deep-learning models are used.**

Design references:
  - Software Architecture § 22 (AI Modules — Preprocessing)
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from bio_oncograph.ai.preprocessing.config import PreprocessingConfig
from bio_oncograph.ai.preprocessing.normalization import normalize_image
from bio_oncograph.ai.preprocessing.report import PreprocessingResult
from bio_oncograph.ai.preprocessing.transforms import (
    bilateral_filter,
    center_crop,
    clahe,
    gamma_correction,
    gaussian_blur,
    histogram_equalize,
    median_filter_op,
    pad_to_size,
    resize,
    unsharp_mask,
)
from bio_oncograph.ai.preprocessing.validation import (
    validate_preprocessing_input,
)

logger = logging.getLogger("BioOncoGraph")

# Type alias for a named transform step
_Step = Tuple[str, Callable[[np.ndarray], np.ndarray]]


class ImagePreprocessingPipeline:
    """
    Configurable, deterministic image preprocessing pipeline.

    The pipeline is constructed from a ``PreprocessingConfig`` via
    ``build()``.  Once built, ``preprocess()`` applies all steps
    sequentially and returns an immutable ``PreprocessingResult``.

    Example::

        pipe = ImagePreprocessingPipeline(
            PreprocessingConfig(target_size=(224, 224))
        ).build()
        result = pipe.preprocess(image)

    Args:
        config: Pipeline configuration.
    """

    def __init__(
        self, config: PreprocessingConfig = PreprocessingConfig()
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            config: Immutable preprocessing configuration.
        """
        self._config = config
        self._steps: List[_Step] = []
        self._built = False

    @property
    def config(self) -> PreprocessingConfig:
        """Return the active configuration."""
        return self._config

    @property
    def is_built(self) -> bool:
        """Return whether ``build()`` has been called."""
        return self._built

    def build(self) -> "ImagePreprocessingPipeline":
        """
        Build the transform pipeline from configuration.

        Must be called before ``preprocess()``.  Calling ``build()``
        again replaces the previous pipeline.

        Returns:
            ``self`` for fluent chaining.
        """
        cfg = self._config
        steps: List[_Step] = []

        # 1. Resize
        steps.append((
            "resize",
            lambda img, ts=cfg.target_size, ip=cfg.interpolation: resize(
                img, ts, ip
            ),
        ))

        # 2. Centre crop (optional)
        if cfg.center_crop is not None:
            steps.append((
                "center_crop",
                lambda img, cs=cfg.center_crop: center_crop(img, cs),
            ))

        # 3. Padding (bring to target_size if crop reduced it)
        steps.append((
            "pad",
            lambda img, ts=cfg.target_size, pm=cfg.padding_mode,
            pv=cfg.padding_value: pad_to_size(img, ts, pm, pv),
        ))

        # 4. Denoising (optional)
        if cfg.denoise_enabled:
            if cfg.denoise_method == "gaussian":
                steps.append((
                    "denoise_gaussian",
                    lambda img, s=cfg.denoise_sigma: gaussian_blur(
                        img, s
                    ),
                ))
            elif cfg.denoise_method == "median":
                steps.append((
                    "denoise_median",
                    lambda img, k=cfg.denoise_kernel_size: median_filter_op(
                        img, k
                    ),
                ))
            elif cfg.denoise_method == "bilateral":
                steps.append((
                    "denoise_bilateral",
                    lambda img, s=cfg.denoise_sigma: bilateral_filter(
                        img, spatial_sigma=s
                    ),
                ))

        # 5. Histogram equalisation (optional, mutually exclusive
        #    with CLAHE)
        if cfg.histogram_equalization and not cfg.clahe_enabled:
            steps.append((
                "histogram_equalization",
                lambda img: histogram_equalize(img),
            ))
        elif cfg.clahe_enabled:
            steps.append((
                "clahe",
                lambda img, cl=cfg.clahe_clip_limit,
                gs=cfg.clahe_grid_size: clahe(img, cl, gs),
            ))

        # 6. Gamma correction (skip if gamma ≈ 1)
        if abs(cfg.gamma - 1.0) > 1e-6:
            steps.append((
                "gamma_correction",
                lambda img, g=cfg.gamma: gamma_correction(img, g),
            ))

        # 7. Sharpening (optional)
        if cfg.sharpen_enabled:
            steps.append((
                "unsharp_mask",
                lambda img, st=cfg.sharpen_strength,
                ss=cfg.sharpen_sigma: unsharp_mask(img, st, ss),
            ))

        # 8. Normalisation (always last)
        steps.append((
            f"normalize_{cfg.normalization_mode}",
            lambda img, m=cfg.normalization_mode, mn=cfg.mean,
            sd=cfg.std: normalize_image(img, m, mn, sd),
        ))

        self._steps = steps
        self._built = True
        logger.info(
            "Pipeline built: %d steps — %s",
            len(steps),
            [s[0] for s in steps],
        )
        return self

    def preprocess(
        self, image: np.ndarray
    ) -> PreprocessingResult:
        """
        Run the preprocessing pipeline on an image.

        Args:
            image: Input image as ``numpy.ndarray``.

        Returns:
            Immutable ``PreprocessingResult``.

        Raises:
            RuntimeError: If ``build()`` has not been called.
            TypeError: If *image* is not an ``ndarray``.
            ValueError: If validation fails.
        """
        if not self._built:
            raise RuntimeError(
                "Pipeline not built. Call build() first."
            )

        cfg = self._config
        validate_preprocessing_input(
            image, cfg.min_resolution, cfg.max_resolution
        )

        original_size = (image.shape[0], image.shape[1])
        warnings: List[str] = []
        applied: List[str] = []

        # Convert to float64 working format
        current = image.astype(np.float64)
        if (
            np.issubdtype(image.dtype, np.floating)
            and image.max() <= 1.0 + 1e-6
        ):
            current = current * 255.0

        for step_name, fn in self._steps:
            current = fn(current)
            applied.append(step_name)

        output_size = (current.shape[0], current.shape[1])

        if output_size != tuple(cfg.target_size):
            warnings.append(
                f"Output size {output_size} differs from "
                f"target {cfg.target_size}."
            )

        metadata: Dict[str, Any] = {
            "original_shape": image.shape,
            "output_shape": current.shape,
            "original_dtype": str(image.dtype),
            "config": {
                "target_size": cfg.target_size,
                "normalization_mode": cfg.normalization_mode,
                "interpolation": cfg.interpolation,
            },
        }

        result = PreprocessingResult(
            processed_image=current,
            applied_transforms=tuple(applied),
            original_size=original_size,
            output_size=output_size,
            warnings=tuple(warnings),
            metadata=metadata,
        )

        logger.info(
            "Preprocessed: %s → %s, %d steps",
            original_size,
            output_size,
            len(applied),
        )
        return result

    def validate(self, image: np.ndarray) -> bool:
        """
        Check whether an image is valid for preprocessing.

        Args:
            image: Input image.

        Returns:
            ``True`` if the image passes all validation checks.
        """
        try:
            cfg = self._config
            validate_preprocessing_input(
                image, cfg.min_resolution, cfg.max_resolution
            )
            return True
        except (TypeError, ValueError):
            return False

    def report(
        self, image: np.ndarray
    ) -> PreprocessingResult:
        """
        Generate a full preprocessing report.

        Alias for ``preprocess()``.

        Args:
            image: Input image.

        Returns:
            Full ``PreprocessingResult``.
        """
        return self.preprocess(image)

    def __repr__(self) -> str:
        steps = [s[0] for s in self._steps] if self._built else []
        return (
            f"ImagePreprocessingPipeline("
            f"built={self._built}, steps={steps})"
        )
