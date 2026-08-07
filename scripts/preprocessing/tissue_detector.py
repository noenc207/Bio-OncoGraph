import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union

class TissueDetector:
    def __init__(self, method: str = 'otsu', hsv_threshold: Tuple[int, int] = (20, 200), 
                 min_object_size: int = 500, morph_kernel_size: int = 5):
        self.method = method
        self.hsv_threshold = hsv_threshold
        self.min_object_size = min_object_size
        self.morph_kernel_size = morph_kernel_size

    def _apply_otsu(self, img_gray: np.ndarray) -> np.ndarray:
        _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    def _apply_hsv(self, img_rgb: np.ndarray) -> np.ndarray:
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_bound = np.array([0, self.hsv_threshold[0], 0])
        upper_bound = np.array([179, self.hsv_threshold[1], 255])
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        return mask

    def _morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _remove_small_objects(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_object_size:
                new_mask[labels == i] = 255
        return new_mask
        
    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(mask, [contour], 0, 255, -1)
        return mask

    def detect(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if self.method == 'otsu':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = self._apply_otsu(img_gray)
        elif self.method == 'hsv':
            mask = self._apply_hsv(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        mask = self._morphological_operations(mask)
        mask = self._remove_small_objects(mask)
        mask = self._fill_holes(mask)
        return mask
