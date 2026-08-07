import cv2
import json
import numpy as np
from PIL import Image
from typing import Dict, Union, Any
import os

class QualityControl:
    def __init__(self, blur_threshold: float = 100.0, max_background_ratio: float = 0.5,
                 pen_mark_threshold: int = 50, bubble_threshold: int = 50):
        self.blur_threshold = blur_threshold
        self.max_background_ratio = max_background_ratio
        self.pen_mark_threshold = pen_mark_threshold
        self.bubble_threshold = bubble_threshold
        self.report = {
            "total_patches": 0,
            "passed": 0,
            "failed_blur": 0,
            "failed_background": 0,
            "failed_pen_marks": 0,
            "failed_bubbles": 0,
            "failed_folded": 0
        }

    def _is_blurry(self, img_gray: np.ndarray) -> bool:
        variance = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def _has_too_much_background(self, img_gray: np.ndarray) -> bool:
        _, bg_mask = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
        bg_ratio = np.sum(bg_mask > 0) / img_gray.size
        return bg_ratio > self.max_background_ratio

    def _has_pen_marks(self, img_hsv: np.ndarray) -> bool:
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_g = cv2.inRange(img_hsv, lower_green, upper_green)
        mask_b = cv2.inRange(img_hsv, lower_blue, upper_blue)
        mask_r1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        
        mask = mask_g | mask_b | mask_r1 | mask_r2
        return np.sum(mask > 0) > self.pen_mark_threshold

    def _has_bubbles(self, img_gray: np.ndarray) -> bool:
        edges = cv2.Canny(img_gray, 100, 200)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=50)
        return circles is not None and len(circles[0]) > 0

    def _has_folded_tissue(self, img_gray: np.ndarray) -> bool:
        _, mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        return lines is not None and len(lines) > 5

    def check_patch(self, image: Union[np.ndarray, Image.Image]) -> bool:
        self.report["total_patches"] += 1
        
        if isinstance(image, Image.Image):
            img_rgb = np.array(image)
        else:
            img_rgb = image

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        if self._has_too_much_background(img_gray):
            self.report["failed_background"] += 1
            return False

        if self._is_blurry(img_gray):
            self.report["failed_blur"] += 1
            return False

        if self._has_pen_marks(img_hsv):
            self.report["failed_pen_marks"] += 1
            return False

        if self._has_bubbles(img_gray):
            self.report["failed_bubbles"] += 1
            return False

        if self._has_folded_tissue(img_gray):
            self.report["failed_folded"] += 1
            return False

        self.report["passed"] += 1
        return True

    def save_report(self, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(self.report, f, indent=4)
