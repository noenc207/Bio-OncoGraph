import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)

class CalibrationInterface(ABC):
    """Interface for probability calibration models."""

    @abstractmethod
    def calibrate(self, predicted_probs: np.ndarray, observed_events: np.ndarray) -> np.ndarray:
        """Calibrates predicted probabilities."""
        pass


class PlattCalibration(CalibrationInterface):
    """Platt scaling (logistic regression) calibration."""

    def calibrate(self, predicted_probs: np.ndarray, observed_events: np.ndarray) -> np.ndarray:
        """Applies mock Platt calibration."""
        logger.info("Applying Platt Calibration.")
        # Stable logit transform
        predicted_probs_safe = np.clip(predicted_probs, 1e-15, 1 - 1e-15)
        logit_probs = np.log(predicted_probs_safe / (1 - predicted_probs_safe))
        X = np.vstack([logit_probs, np.ones(len(logit_probs))]).T
        
        # Mock optimization via least squares
        weights, _, _, _ = np.linalg.lstsq(X, observed_events, rcond=None)
        calibrated = 1.0 / (1.0 + np.exp(-(X @ weights)))
        return np.clip(calibrated, 0.0, 1.0)


class IsotonicCalibration(CalibrationInterface):
    """Isotonic regression calibration using simplified PAVA."""

    def calibrate(self, predicted_probs: np.ndarray, observed_events: np.ndarray) -> np.ndarray:
        """Applies isotonic regression calibration."""
        logger.info("Applying Isotonic Calibration.")
        order = np.argsort(predicted_probs)
        y_sorted = observed_events[order].astype(float)
        n = len(y_sorted)
        
        calibrated_sorted = y_sorted.copy()
        
        # Simplified PAVA mock
        for _ in range(3):  # Multiple passes for stability
            for i in range(n - 1):
                if calibrated_sorted[i] > calibrated_sorted[i + 1]:
                    avg = (calibrated_sorted[i] + calibrated_sorted[i + 1]) / 2.0
                    calibrated_sorted[i:i+2] = avg
                    
        calibrated = np.zeros_like(calibrated_sorted)
        calibrated[order] = calibrated_sorted
        return np.clip(calibrated, 0.0, 1.0)


class CalibrationMetrics:
    """Metrics for evaluating calibration."""

    def brier_score(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Computes the Brier score."""
        return float(np.mean((predicted - observed) ** 2))

    def calibration_slope(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        """Computes the calibration slope."""
        predicted_safe = np.clip(predicted, 1e-15, 1 - 1e-15)
        logit_p = np.log(predicted_safe / (1 - predicted_safe))
        X = np.vstack([logit_p, np.ones(len(logit_p))]).T
        slope, _ = np.linalg.lstsq(X, observed, rcond=None)[0]
        return float(slope)

    def hosmer_lemeshow(self, predicted: np.ndarray, observed: np.ndarray, num_bins: int = 10) -> float:
        """Computes the Hosmer-Lemeshow goodness-of-fit statistic."""
        order = np.argsort(predicted)
        predicted_sorted = predicted[order]
        observed_sorted = observed[order]

        bins = np.array_split(np.arange(len(predicted)), num_bins)
        hl_stat = 0.0

        for bin_idx in bins:
            if len(bin_idx) == 0:
                continue
            obs_events = np.sum(observed_sorted[bin_idx])
            exp_events = np.sum(predicted_sorted[bin_idx])
            n_bin = len(bin_idx)
            
            denom = exp_events * (1 - exp_events / n_bin)
            if denom > 1e-7:
                hl_stat += ((obs_events - exp_events) ** 2) / denom

        return float(hl_stat)
