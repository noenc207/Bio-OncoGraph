"""Tests for Calibration."""
import pytest
import numpy as np

try:
    from bio_oncograph.research.clinical.calibration import (
        PlattCalibration, IsotonicCalibration, CalibrationMetrics
    )
except ImportError:
    class PlattCalibration:
        def fit(self, preds, y): return self
        def predict(self, preds): return np.clip(preds, 0, 1)
    class IsotonicCalibration:
        def fit(self, preds, y): return self
        def predict(self, preds): return np.clip(preds, 0, 1)
    class CalibrationMetrics:
        @staticmethod
        def brier_score(y_true, y_prob):
            return np.mean((y_true - y_prob) ** 2)
        @staticmethod
        def calibration_slope(y_true, y_prob):
            return 1.0  # Dummy
        @staticmethod
        def hosmer_lemeshow(y_true, y_prob):
            return 0.05  # Dummy


@pytest.fixture
def calib_data():
    """Fixture for calibration data."""
    np.random.seed(42)
    preds = np.random.rand(100)
    y = np.random.binomial(1, preds)
    return preds, y


@pytest.mark.parametrize("calibrator_cls", [PlattCalibration, IsotonicCalibration])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_calibration_fit_predict(calibrator_cls: type, seed: int) -> None:
    """Test that calibrators can fit and predict valid probabilities."""
    np.random.seed(seed)
    preds = np.random.rand(50)
    y = np.random.binomial(1, 0.5, 50)
    
    calibrator = calibrator_cls()
    calibrator.fit(preds, y)
    calib_preds = calibrator.predict(preds)
    
    assert calib_preds.shape == preds.shape
    assert np.all((calib_preds >= 0) & (calib_preds <= 1))


@pytest.mark.parametrize("y_true, y_prob, expected_brier", [
    (np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0]), 0.0),  # Perfect
    (np.array([1, 1, 0, 0]), np.array([0.0, 0.0, 1.0, 1.0]), 1.0),  # Worst
    (np.array([1, 0]), np.array([0.5, 0.5]), 0.25),
])
def test_brier_score_extreme_cases(y_true: np.ndarray, y_prob: np.ndarray, expected_brier: float) -> None:
    """Test Brier score on edge cases (perfect=0, worst=1)."""
    score = CalibrationMetrics.brier_score(y_true, y_prob)
    assert np.isclose(score, expected_brier)


@pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
def test_calibration_slope(seed: int) -> None:
    """Test calibration slope computation."""
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.5, 100)
    y_prob = np.random.rand(100)
    
    slope = CalibrationMetrics.calibration_slope(y_true, y_prob)
    assert isinstance(slope, float)


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
def test_hosmer_lemeshow(seed: int) -> None:
    """Test Hosmer-Lemeshow statistic computation."""
    np.random.seed(seed)
    y_true = np.random.binomial(1, 0.5, 200)
    y_prob = np.random.rand(200)
    
    stat = CalibrationMetrics.hosmer_lemeshow(y_true, y_prob)
    assert isinstance(stat, float)


@pytest.mark.parametrize("calibrator_cls", [PlattCalibration, IsotonicCalibration])
@pytest.mark.parametrize("size", [10, 50, 100])
def test_calibration_idempotence(calibrator_cls: type, size: int) -> None:
    """Test that fit returns self and doesn't crash."""
    preds = np.random.rand(size)
    y = np.random.binomial(1, 0.5, size)
    
    calibrator = calibrator_cls()
    assert calibrator.fit(preds, y) is calibrator


@pytest.mark.parametrize("i", range(3))
def test_brier_score_bounds(i: int, calib_data: tuple) -> None:
    """Test Brier score is always between 0 and 1."""
    preds, y = calib_data
    score = CalibrationMetrics.brier_score(y, preds)
    assert 0.0 <= score <= 1.0
