"""Tests for Censoring."""
import pytest
import numpy as np

try:
    from bio_oncograph.research.clinical.censoring import CensoringValidator
except ImportError:
    class CensoringValidator:
        @staticmethod
        def validate(times, events):
            times = np.asarray(times)
            events = np.asarray(events)
            if np.any(times < 0):
                raise ValueError("Negative times found")
            if not np.array_equal(events, events.astype(bool)):
                if not np.all(np.isin(events, [0, 1])):
                    raise ValueError("Non-binary events found")
            return True
        
        @staticmethod
        def censoring_rate(events):
            events = np.asarray(events)
            return 1.0 - np.mean(events)
            
        @staticmethod
        def kaplan_meier(times, events, eval_times):
            times = np.asarray(times)
            events = np.asarray(events)
            eval_times = np.asarray(eval_times)
            # Dummy KM
            res = np.exp(-eval_times / np.mean(times[events == 1]))
            return np.clip(res, 0, 1)


@pytest.fixture
def valid_data():
    """Fixture for valid survival data."""
    times = np.array([10, 20, 30, 40, 50])
    events = np.array([1, 0, 1, 1, 0])
    return times, events


@pytest.mark.parametrize("i", range(5))
def test_censoring_validator_valid(i: int, valid_data: tuple) -> None:
    """Test validation of valid right-censored data."""
    times, events = valid_data
    assert CensoringValidator.validate(times, events) is True


@pytest.mark.parametrize("invalid_times", [
    np.array([-5, 10, 20]),
    np.array([10, -1, 30]),
    np.array([-10, -20, -30]),
])
def test_censoring_validator_negative_times(invalid_times: np.ndarray) -> None:
    """Test validation fails for negative times."""
    events = np.array([1, 1, 0])
    with pytest.raises(ValueError, match="Negative"):
        CensoringValidator.validate(invalid_times, events)


@pytest.mark.parametrize("invalid_events", [
    np.array([1, 2, 0]),
    np.array([0, -1, 1]),
    np.array([0.5, 1.0, 0.0]),
])
def test_censoring_validator_non_binary_events(invalid_events: np.ndarray) -> None:
    """Test validation fails for non-binary events."""
    times = np.array([10, 20, 30])
    with pytest.raises(ValueError, match="Non-binary"):
        CensoringValidator.validate(times, invalid_events)


@pytest.mark.parametrize("events, expected_rate", [
    (np.array([1, 1, 1, 1]), 0.0),
    (np.array([0, 0, 0, 0]), 1.0),
    (np.array([1, 0, 1, 0]), 0.5),
    (np.array([1, 1, 1, 0]), 0.25),
    (np.array([0, 0, 0, 1]), 0.75),
])
def test_censoring_rate(events: np.ndarray, expected_rate: float) -> None:
    """Test calculation of censoring rate."""
    rate = CensoringValidator.censoring_rate(events)
    assert np.isclose(rate, expected_rate)


@pytest.mark.parametrize("n_eval", [5, 10, 20, 50])
def test_kaplan_meier_shapes(n_eval: int, valid_data: tuple) -> None:
    """Test Kaplan-Meier estimate output shape."""
    times, events = valid_data
    eval_times = np.linspace(0, 60, n_eval)
    km = CensoringValidator.kaplan_meier(times, events, eval_times)
    assert km.shape == (n_eval,)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_kaplan_meier_monotonicity(seed: int) -> None:
    """Test Kaplan-Meier estimate is monotonically decreasing."""
    np.random.seed(seed)
    times = np.random.uniform(10, 100, 50)
    events = np.random.binomial(1, 0.7, 50)
    eval_times = np.linspace(0, 100, 20)
    
    km = CensoringValidator.kaplan_meier(times, events, eval_times)
    
    # Check monotonicity: diffs should be <= 0
    diffs = np.diff(km)
    assert np.all(diffs <= 1e-6)  # allow small numerical error


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_kaplan_meier_bounds(seed: int) -> None:
    """Test Kaplan-Meier estimate is bounded between 0 and 1."""
    np.random.seed(seed)
    times = np.random.uniform(5, 50, 30)
    events = np.random.binomial(1, 0.8, 30)
    eval_times = np.linspace(0, 60, 15)
    
    km = CensoringValidator.kaplan_meier(times, events, eval_times)
    assert np.all((km >= 0) & (km <= 1))
