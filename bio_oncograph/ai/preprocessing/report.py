"""Preprocessing result container."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True, eq=False)
class PreprocessingResult:
    """Immutable result of image preprocessing.

    Attributes:
        processed_image: The preprocessed image array.
        applied_transforms: Names of transforms applied, in order.
        original_size: ``(height, width)`` of the input.
        output_size: ``(height, width)`` of the output.
        warnings: Human-readable warnings.
        metadata: Auxiliary information.
    """

    processed_image: np.ndarray
    applied_transforms: Tuple[str, ...]
    original_size: Tuple[int, int]
    output_size: Tuple[int, int]
    warnings: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PreprocessingResult):
            return NotImplemented
        return (
            self.applied_transforms == other.applied_transforms
            and self.original_size == other.original_size
            and self.output_size == other.output_size
        )

    def __hash__(self) -> int:
        return hash(
            (self.applied_transforms, self.original_size, self.output_size)
        )
