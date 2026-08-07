"""Utility functions for WSI processing and patch extraction."""

from typing import List, Tuple, Optional
import logging

import numpy as np

from bio_oncograph.infrastructure.wsi.roi import ROI

logger = logging.getLogger(__name__)


def compute_grid_coordinates(
    width: int,
    height: int,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    roi: Optional[ROI] = None
) -> List[Tuple[int, int]]:
    """Compute top-left coordinates for a grid of patches.

    Args:
        width: Total width of the area to grid.
        height: Total height of the area to grid.
        patch_size: (width, height) of each patch.
        stride: (stride_x, stride_y) distance between patches.
        roi: Optional ROI to filter coordinates. If provided, the center
            of the patch must fall within the ROI.

    Returns:
        A list of (x, y) coordinates for the patches.
    """
    coords = []
    pw, ph = patch_size
    sx, sy = stride

    for y in range(0, height - ph + 1, sy):
        for x in range(0, width - pw + 1, sx):
            if roi is not None:
                center_x = x + pw / 2.0
                center_y = y + ph / 2.0
                if not roi.contains(center_x, center_y):
                    continue
            coords.append((x, y))

    return coords


def estimate_tissue_fraction(patch_array: np.ndarray, threshold: int = 210) -> float:
    """Estimate the fraction of tissue in an RGB patch.

    A simple heuristic that assumes background is bright (near white).

    Args:
        patch_array: An RGB numpy array of shape (H, W, 3).
        threshold: Grayscale intensity threshold (0-255) above which a pixel
            is considered background.

    Returns:
        The fraction of tissue pixels [0.0, 1.0].
    """
    if patch_array.size == 0:
        return 0.0
        
    if patch_array.shape[-1] == 4:
        # Convert RGBA to RGB simply by dropping alpha
        patch_array = patch_array[..., :3]
        
    # Convert to grayscale using simple average
    gray = np.mean(patch_array, axis=-1)
    
    # Tissue is darker than the threshold
    tissue_mask = gray < threshold
    
    return float(np.sum(tissue_mask)) / gray.size


def patch_to_rgb(patch_array: np.ndarray) -> np.ndarray:
    """Ensure a patch is in RGB format.

    Args:
        patch_array: The input numpy array (grayscale, RGB, or RGBA).

    Returns:
        An RGB numpy array of shape (H, W, 3).
        
    Raises:
        ValueError: If the array has an unexpected shape.
    """
    if patch_array.ndim == 2:
        # Grayscale to RGB
        return np.stack((patch_array,) * 3, axis=-1)
        
    if patch_array.ndim == 3:
        channels = patch_array.shape[-1]
        if channels == 3:
            return patch_array
        if channels == 4:
            # RGBA to RGB (drop alpha)
            return patch_array[..., :3]
        if channels == 1:
            # Single channel with trailing dimension
            return np.concatenate([patch_array] * 3, axis=-1)
            
    raise ValueError(f"Unexpected patch shape: {patch_array.shape}")
