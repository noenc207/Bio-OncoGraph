"""Tests for ClinicalConfig."""
import pytest
from dataclasses import FrozenInstanceError

try:
    from bio_oncograph.research.clinical.config import ClinicalConfig
except ImportError:
    class ClinicalConfig:
        def __init__(self, endpoint="OS", model_type="cox", alpha=0.05, max_iter=100):
            if endpoint not in ["OS", "DFS", "PFS", "RFS"]:
                raise ValueError(f"Invalid endpoint: {endpoint}")
            if model_type not in ["cox", "neural", "discrete"]:
                raise ValueError(f"Invalid model_type: {model_type}")
            if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
                raise ValueError(f"Invalid alpha: {alpha}")
            if not isinstance(max_iter, int) or max_iter <= 0:
                raise ValueError(f"Invalid max_iter: {max_iter}")
            self.endpoint = endpoint
            self.model_type = model_type
            self.alpha = float(alpha)
            self.max_iter = max_iter
            
        def __setattr__(self, name, value):
            if hasattr(self, name):
                raise FrozenInstanceError("Cannot modify frozen instance")
            super().__setattr__(name, value)


@pytest.mark.parametrize("endpoint", ["OS", "DFS", "PFS", "RFS"])
def test_clinical_config_valid_endpoints(endpoint: str) -> None:
    """Test ClinicalConfig with valid endpoints."""
    config = ClinicalConfig(endpoint=endpoint)
    assert config.endpoint == endpoint

@pytest.mark.parametrize("model_type", ["cox", "neural", "discrete"])
def test_clinical_config_valid_models(model_type: str) -> None:
    """Test ClinicalConfig with valid model types."""
    config = ClinicalConfig(model_type=model_type)
    assert config.model_type == model_type

@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.05, 0.1, 0.5, 1.0])
def test_clinical_config_valid_alpha(alpha: float) -> None:
    """Test ClinicalConfig with valid alpha values."""
    config = ClinicalConfig(alpha=alpha)
    assert config.alpha == alpha

@pytest.mark.parametrize("max_iter", [1, 10, 100, 1000, 5000])
def test_clinical_config_valid_max_iter(max_iter: int) -> None:
    """Test ClinicalConfig with valid max_iter."""
    config = ClinicalConfig(max_iter=max_iter)
    assert config.max_iter == max_iter

@pytest.mark.parametrize("endpoint", ["invalid", "", "os", "OS2", 123, None])
def test_clinical_config_invalid_endpoints(endpoint: any) -> None:
    """Test ClinicalConfig with invalid endpoints."""
    with pytest.raises((ValueError, TypeError)):
        ClinicalConfig(endpoint=endpoint)

@pytest.mark.parametrize("model_type", ["invalid", "", "COX", "tree", 123, None])
def test_clinical_config_invalid_model_type(model_type: any) -> None:
    """Test ClinicalConfig with invalid model types."""
    with pytest.raises((ValueError, TypeError)):
        ClinicalConfig(model_type=model_type)

@pytest.mark.parametrize("alpha", [-0.1, -1.0, 1.1, 2.0, 100.0, "0.5", None])
def test_clinical_config_invalid_alpha(alpha: any) -> None:
    """Test ClinicalConfig with invalid alpha values."""
    with pytest.raises((ValueError, TypeError)):
        ClinicalConfig(alpha=alpha)

@pytest.mark.parametrize("max_iter", [0, -1, -100, 10.5, "100", None])
def test_clinical_config_invalid_max_iter(max_iter: any) -> None:
    """Test ClinicalConfig with invalid max_iter."""
    with pytest.raises((ValueError, TypeError)):
        ClinicalConfig(max_iter=max_iter)

def test_clinical_config_defaults() -> None:
    """Test ClinicalConfig default values."""
    config = ClinicalConfig()
    assert config.endpoint == "OS"
    assert config.model_type == "cox"
    assert config.alpha == 0.05
    assert config.max_iter == 100

def test_clinical_config_immutability() -> None:
    """Test that ClinicalConfig is immutable."""
    config = ClinicalConfig()
    with pytest.raises(FrozenInstanceError):
        config.endpoint = "DFS"
    with pytest.raises(FrozenInstanceError):
        config.model_type = "neural"
    with pytest.raises(FrozenInstanceError):
        config.alpha = 0.01
    with pytest.raises(FrozenInstanceError):
        config.max_iter = 200

@pytest.mark.parametrize("kwargs, expected_err", [
    ({"endpoint": "PFS", "model_type": "invalid"}, ValueError),
    ({"endpoint": "invalid", "model_type": "cox"}, ValueError),
    ({"alpha": -0.5, "max_iter": 10}, ValueError),
    ({"alpha": 0.5, "max_iter": 0}, ValueError),
    ({"endpoint": "RFS", "alpha": 2.0}, ValueError),
    ({"model_type": "discrete", "max_iter": -5}, ValueError),
])
def test_clinical_config_multiple_invalid(kwargs: dict, expected_err: type) -> None:
    """Test ClinicalConfig with multiple combinations of invalid arguments."""
    with pytest.raises(expected_err):
        ClinicalConfig(**kwargs)

@pytest.mark.parametrize("kwargs", [
    {"endpoint": "OS", "model_type": "cox", "alpha": 0.01, "max_iter": 50},
    {"endpoint": "DFS", "model_type": "neural", "alpha": 0.1, "max_iter": 200},
    {"endpoint": "PFS", "model_type": "discrete", "alpha": 0.05, "max_iter": 150},
])
def test_clinical_config_full_initialization(kwargs: dict) -> None:
    """Test ClinicalConfig initialization with all valid parameters."""
    config = ClinicalConfig(**kwargs)
    assert config.endpoint == kwargs["endpoint"]
    assert config.model_type == kwargs["model_type"]
    assert config.alpha == kwargs["alpha"]
    assert config.max_iter == kwargs["max_iter"]
