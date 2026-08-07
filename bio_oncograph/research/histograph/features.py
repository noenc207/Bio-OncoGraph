"""Feature extractors for histopathology graph construction."""
import abc
import logging

import numpy as np

logger = logging.getLogger(__name__)

class FeatureExtractor(abc.ABC):
    """Abstract base class for feature extractors."""

    @abc.abstractmethod
    def extract(self, patch: np.ndarray) -> np.ndarray:
        """Extracts features from an image patch.

        Args:
            patch: Numpy array representing the image patch.

        Returns:
            Numpy array of extracted features.
        """
        pass

class MorphologyFeatureExtractor(FeatureExtractor):
    """Extracts morphology features from a binary mask."""

    def extract(self, patch: np.ndarray) -> np.ndarray:
        """Extracts morphology features (area, perimeter_approx, aspect_ratio, solidity).

        Args:
            patch: Binary mask numpy array of shape (H, W).

        Returns:
            Numpy array of morphology features.
        """
        if patch.ndim != 2:
            logger.warning("Morphology extraction expects a 2D binary mask.")
            
        # Basic area
        area = np.sum(patch)
        
        if area == 0:
            return np.zeros(4, dtype=np.float32)

        # Simple perimeter approximation (edge pixels)
        padded = np.pad(patch, 1, mode='constant')
        edges = np.abs(np.diff(padded, axis=0))[:, :-1] + np.abs(np.diff(padded, axis=1))[:-1, :]
        perimeter_approx = np.sum(edges > 0)
        
        # Bounding box for aspect ratio and solidity
        y_indices, x_indices = np.nonzero(patch)
        if len(y_indices) > 0 and len(x_indices) > 0:
            h = np.max(y_indices) - np.min(y_indices) + 1
            w = np.max(x_indices) - np.min(x_indices) + 1
            aspect_ratio = float(w) / float(h)
            
            # Simple solidity approx
            convex_area = h * w
            solidity = area / convex_area if convex_area > 0 else 0.0
        else:
            aspect_ratio = 0.0
            solidity = 0.0

        return np.array([area, perimeter_approx, aspect_ratio, solidity], dtype=np.float32)

class IntensityFeatureExtractor(FeatureExtractor):
    """Extracts intensity features from an image patch."""

    def extract(self, patch: np.ndarray) -> np.ndarray:
        """Extracts intensity features (mean, std, min, max, median) per channel.

        Args:
            patch: Image numpy array of shape (H, W, C) or (H, W).

        Returns:
            Numpy array of intensity features.
        """
        if patch.ndim == 2:
            patch = patch[..., np.newaxis]
            
        c = patch.shape[2]
        features = []
        
        for i in range(c):
            channel = patch[:, :, i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)
            median_val = np.median(channel)
            
            features.extend([mean_val, std_val, min_val, max_val, median_val])
            
        return np.array(features, dtype=np.float32)

class TextureFeatureExtractor(FeatureExtractor):
    """Extracts texture features from a grayscale image patch."""

    def extract(self, patch: np.ndarray) -> np.ndarray:
        """Extracts texture features (variance, gradient magnitude mean).

        Args:
            patch: Grayscale image numpy array of shape (H, W).

        Returns:
            Numpy array of texture features.
        """
        if patch.ndim != 2:
            logger.warning("Texture extraction expects a 2D grayscale image.")
            
        variance = np.var(patch)
        
        # Simple gradient magnitude
        gx = np.gradient(patch, axis=1)
        gy = np.gradient(patch, axis=0)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag_mean = np.mean(grad_mag)
        
        return np.array([variance, grad_mag_mean], dtype=np.float32)
