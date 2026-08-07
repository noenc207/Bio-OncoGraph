import pytest
import numpy as np

class ReproducibilityValidator:
    """Validates reproducibility of deterministic operations."""
    def __init__(self, seed: int = 42):
        """Initializes with a seed."""
        self.seed = seed
        
    def run_deterministic(self, size: int) -> np.ndarray:
        """Runs a deterministic random generation."""
        rng = np.random.RandomState(self.seed)
        return rng.rand(size)
        
    def run_stochastic(self, size: int) -> np.ndarray:
        """Runs a stochastic random generation."""
        rng = np.random.RandomState()
        return rng.rand(size)
        
    def check_consistency(self, run1: np.ndarray, run2: np.ndarray) -> bool:
        """Checks if two arrays are strictly equal."""
        return np.array_equal(run1, run2)

@pytest.fixture
def validator() -> ReproducibilityValidator:
    """Fixture providing a ReproducibilityValidator."""
    return ReproducibilityValidator(seed=123)

def test_reproducibility_same_instance(validator: ReproducibilityValidator) -> None:
    """Test reproducibility calling the same method twice."""
    run1 = validator.run_deterministic(100)
    run2 = validator.run_deterministic(100)
    assert validator.check_consistency(run1, run2)

def test_reproducibility_different_instances() -> None:
    """Test reproducibility across different instances with same seed."""
    v1 = ReproducibilityValidator(seed=42)
    v2 = ReproducibilityValidator(seed=42)
    assert v1.check_consistency(v1.run_deterministic(50), v2.run_deterministic(50))

def test_reproducibility_different_seeds() -> None:
    """Test reproducibility fails across different instances with diff seeds."""
    v1 = ReproducibilityValidator(seed=42)
    v2 = ReproducibilityValidator(seed=43)
    assert not v1.check_consistency(v1.run_deterministic(50), v2.run_deterministic(50))

def test_reproducibility_stochastic_fails(validator: ReproducibilityValidator) -> None:
    """Test that stochastic runs are not equal."""
    run1 = validator.run_stochastic(100)
    run2 = validator.run_stochastic(100)
    assert not validator.check_consistency(run1, run2)

def test_reproducibility_check_consistency_true(validator: ReproducibilityValidator) -> None:
    """Test check_consistency explicitly with equal arrays."""
    arr = np.array([1.0, 2.0, 3.0])
    assert validator.check_consistency(arr, arr.copy())

def test_reproducibility_check_consistency_false(validator: ReproducibilityValidator) -> None:
    """Test check_consistency explicitly with unequal arrays."""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.0, 2.0, 3.1])
    assert not validator.check_consistency(arr1, arr2)

def test_reproducibility_check_consistency_diff_shape(validator: ReproducibilityValidator) -> None:
    """Test check_consistency with different shapes."""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([1.0, 2.0])
    assert not validator.check_consistency(arr1, arr2)

def test_reproducibility_large_size(validator: ReproducibilityValidator) -> None:
    """Test reproducibility with a large array."""
    run1 = validator.run_deterministic(10000)
    run2 = validator.run_deterministic(10000)
    assert validator.check_consistency(run1, run2)

def test_reproducibility_zero_size(validator: ReproducibilityValidator) -> None:
    """Test reproducibility with size zero."""
    run1 = validator.run_deterministic(0)
    run2 = validator.run_deterministic(0)
    assert validator.check_consistency(run1, run2)
    
def test_reproducibility_seed_0() -> None:
    """Test reproducibility with seed 0."""
    v1 = ReproducibilityValidator(seed=0)
    v2 = ReproducibilityValidator(seed=0)
    assert v1.check_consistency(v1.run_deterministic(10), v2.run_deterministic(10))

def test_reproducibility_seed_negative() -> None:
    """Test reproducibility with negative seed wrapping to valid domain or failing gracefully."""
    with pytest.raises(ValueError):
        np.random.RandomState(-1)

def test_reproducibility_seed_large() -> None:
    """Test reproducibility with a large seed."""
    v1 = ReproducibilityValidator(seed=999999)
    v2 = ReproducibilityValidator(seed=999999)
    assert v1.check_consistency(v1.run_deterministic(10), v2.run_deterministic(10))

def test_reproducibility_run1_type(validator: ReproducibilityValidator) -> None:
    """Test output type."""
    run1 = validator.run_deterministic(10)
    assert isinstance(run1, np.ndarray)

def test_reproducibility_run1_shape(validator: ReproducibilityValidator) -> None:
    """Test output shape."""
    run1 = validator.run_deterministic(15)
    assert run1.shape == (15,)

def test_reproducibility_run1_dtype(validator: ReproducibilityValidator) -> None:
    """Test output dtype."""
    run1 = validator.run_deterministic(10)
    assert run1.dtype == np.float64

def test_reproducibility_bound_min(validator: ReproducibilityValidator) -> None:
    """Test min bound of generated array."""
    run1 = validator.run_deterministic(100)
    assert np.min(run1) >= 0.0

def test_reproducibility_bound_max(validator: ReproducibilityValidator) -> None:
    """Test max bound of generated array."""
    run1 = validator.run_deterministic(100)
    assert np.max(run1) < 1.0

def test_reproducibility_stochastic_shape(validator: ReproducibilityValidator) -> None:
    """Test stochastic run shape."""
    run1 = validator.run_stochastic(20)
    assert run1.shape == (20,)

def test_reproducibility_stochastic_type(validator: ReproducibilityValidator) -> None:
    """Test stochastic run type."""
    run1 = validator.run_stochastic(20)
    assert isinstance(run1, np.ndarray)

def test_reproducibility_consistency_empty() -> None:
    """Test consistency with empty arrays."""
    v = ReproducibilityValidator()
    assert v.check_consistency(np.array([]), np.array([]))

def test_reproducibility_consistency_nan() -> None:
    """Test consistency with NaN arrays."""
    v = ReproducibilityValidator()
    # np.array_equal returns False for NaNs
    assert not v.check_consistency(np.array([np.nan]), np.array([np.nan]))
