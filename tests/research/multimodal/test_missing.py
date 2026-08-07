import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# ==============================================================================
# MOCKS
# ==============================================================================
class MissingDataHandler(ABC):
    @abstractmethod
    def handle(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass

class MaskingHandler(MissingDataHandler):
    def handle(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed = {}
        for k, v in data.items():
            if v.size == 0 or np.isnan(v).all():
                processed[k] = np.zeros_like(v) if v.size > 0 else np.zeros((1,))
            else:
                processed[k] = np.where(np.isnan(v), 0, v)
        return processed

class ImputationHandler(MissingDataHandler):
    def handle(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed = {}
        for k, v in data.items():
            if v.size == 0 or np.isnan(v).all():
                processed[k] = np.zeros_like(v) if v.size > 0 else np.zeros((1,))
            else:
                mean_val = np.nanmean(v)
                processed[k] = np.where(np.isnan(v), mean_val, v)
        return processed

class DropHandler(MissingDataHandler):
    def handle(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed = {}
        for k, v in data.items():
            if v.size > 0 and not np.isnan(v).any():
                processed[k] = v
        return processed

# ==============================================================================
# TESTS
# ==============================================================================

@pytest.fixture
def sample_data_with_missing() -> Dict[str, np.ndarray]:
    return {
        "mod1": np.array([1.0, 2.0, np.nan, 4.0]),
        "mod2": np.array([np.nan, np.nan, np.nan]),
        "mod3": np.array([]),
        "mod4": np.array([1.0, 2.0, 3.0])
    }

def test_masking_handler_zero_fill(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test MaskingHandler replaces NaNs with 0."""
    handler = MaskingHandler()
    res = handler.handle(sample_data_with_missing)
    np.testing.assert_array_equal(res["mod1"], [1.0, 2.0, 0.0, 4.0])

def test_masking_handler_all_nan(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test MaskingHandler handles arrays with all NaNs."""
    handler = MaskingHandler()
    res = handler.handle(sample_data_with_missing)
    np.testing.assert_array_equal(res["mod2"], [0.0, 0.0, 0.0])

def test_masking_handler_empty(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test MaskingHandler handles empty arrays."""
    handler = MaskingHandler()
    res = handler.handle(sample_data_with_missing)
    assert res["mod3"].shape == (1,)
    assert res["mod3"][0] == 0.0

def test_imputation_handler_mean_fill(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test ImputationHandler replaces NaNs with mean."""
    handler = ImputationHandler()
    res = handler.handle(sample_data_with_missing)
    mean_val = np.mean([1.0, 2.0, 4.0])
    np.testing.assert_allclose(res["mod1"], [1.0, 2.0, mean_val, 4.0])

def test_imputation_handler_all_nan(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test ImputationHandler handles arrays with all NaNs."""
    handler = ImputationHandler()
    res = handler.handle(sample_data_with_missing)
    np.testing.assert_array_equal(res["mod2"], [0.0, 0.0, 0.0])

def test_drop_handler_drops_nan_keys(sample_data_with_missing: Dict[str, np.ndarray]) -> None:
    """Test DropHandler removes keys with any NaNs or empty arrays."""
    handler = DropHandler()
    res = handler.handle(sample_data_with_missing)
    assert "mod1" not in res
    assert "mod2" not in res
    assert "mod3" not in res
    assert "mod4" in res
    np.testing.assert_array_equal(res["mod4"], [1.0, 2.0, 3.0])

@pytest.mark.parametrize("handler_cls", [
    MaskingHandler, ImputationHandler, DropHandler
])
def test_handlers_empty_dict(handler_cls: type) -> None:
    """Test handlers with empty dictionary input."""
    handler = handler_cls()
    res = handler.handle({})
    assert res == {}

@pytest.mark.parametrize("handler_cls", [
    MaskingHandler, ImputationHandler, DropHandler
])
def test_handlers_no_missing_data(handler_cls: type) -> None:
    """Test handlers leave perfectly clean data intact."""
    handler = handler_cls()
    data = {"clean": np.array([1.0, 2.0, 3.0])}
    res = handler.handle(data)
    np.testing.assert_array_equal(res["clean"], data["clean"])

@pytest.mark.parametrize("seed", range(5))
def test_imputation_randomized_data(seed: int) -> None:
    """Test imputation consistency with random data."""
    np.random.seed(seed)
    data = {"rand": np.random.randn(100)}
    data["rand"][np.random.choice(100, 10, replace=False)] = np.nan
    handler = ImputationHandler()
    res = handler.handle(data)
    assert not np.isnan(res["rand"]).any()

@pytest.mark.parametrize("mod_count", [1, 2, 5, 10])
def test_drop_handler_multimodality(mod_count: int) -> None:
    """Test DropHandler correctly processes varying number of modalities."""
    data = {f"mod{i}": np.array([float(i)]) for i in range(mod_count)}
    data["nan_mod"] = np.array([np.nan])
    handler = DropHandler()
    res = handler.handle(data)
    assert len(res) == mod_count
    assert "nan_mod" not in res
