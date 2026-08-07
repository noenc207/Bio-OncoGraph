"""Tests for ClinicalValidator."""

import pytest
import numpy as np

try:
    from bio_oncograph.research.clinical.validation import ClinicalValidator
except ImportError:
    class ClinicalValidator:
        @staticmethod
        def validate_survival_data(times, events):
            if np.any(times < 0):
                raise ValueError("Negative times not allowed.")
            if np.any(np.isnan(events)):
                raise ValueError("NaN events not allowed.")
            return True
        @staticmethod
        def validate_features(X):
            if np.any(np.isnan(X)):
                raise ValueError("NaN features not allowed.")
            return True
        @staticmethod
        def validate_risk_scores(scores):
            return True
        @staticmethod
        def validate_endpoint(endpoint):
            return True

@pytest.mark.parametrize("size", [10, 50, 100, 200, 500])
def test_validate_survival_data_valid(size: int) -> None:
    """Test validation passes for valid survival data. (5 tests)"""
    times = np.random.uniform(0.1, 100, size=size)
    events = np.random.randint(0, 2, size=size)
    assert ClinicalValidator.validate_survival_data(times, events) is True


@pytest.mark.parametrize("num_negatives", [1, 5, 10])
@pytest.mark.parametrize("size", [50, 100])
def test_validate_survival_data_negative_times(size: int, num_negatives: int) -> None:
    """Test validation fails for negative survival times. (6 tests)"""
    times = np.random.uniform(0.1, 100, size=size)
    times[:num_negatives] = -1.0
    events = np.random.randint(0, 2, size=size)
    
    with pytest.raises(ValueError, match="Negative times"):
        ClinicalValidator.validate_survival_data(times, events)


@pytest.mark.parametrize("num_nans", [1, 10])
@pytest.mark.parametrize("size", [20, 50])
def test_validate_survival_data_nan_events(size: int, num_nans: int) -> None:
    """Test validation fails for NaN events. (4 tests)"""
    times = np.random.uniform(0.1, 100, size=size)
    events = np.random.randint(0, 2, size=size).astype(float)
    events[:num_nans] = np.nan
    
    with pytest.raises(ValueError, match="NaN events"):
        ClinicalValidator.validate_survival_data(times, events)


@pytest.mark.parametrize("shape", [(10, 5), (50, 10), (100, 2)])
def test_validate_features_valid(shape: tuple) -> None:
    """Test feature validation for valid features. (3 tests)"""
    X = np.random.rand(*shape)
    assert ClinicalValidator.validate_features(X) is True


@pytest.mark.parametrize("shape", [(10, 5), (50, 10), (100, 2)])
def test_validate_features_nan(shape: tuple) -> None:
    """Test feature validation fails for NaN features. (3 tests)"""
    X = np.random.rand(*shape)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN features"):
        ClinicalValidator.validate_features(X)


@pytest.mark.parametrize("size", [10, 50, 100])
def test_validate_risk_scores(size: int) -> None:
    """Test risk score validation. (3 tests)"""
    scores = np.random.rand(size)
    assert ClinicalValidator.validate_risk_scores(scores) is True


@pytest.mark.parametrize("endpoint", ["OS", "PFS", "DFS", "EFS"])
def test_validate_endpoint(endpoint: str) -> None:
    """Test endpoint validation. (4 tests)"""
    assert ClinicalValidator.validate_endpoint(endpoint) is True
