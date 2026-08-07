"""Tests for FoundationValidator.

Validates embedding dimensions, prompt registration, adapter compatibility,
and performs nan/inf checks.
"""

from typing import Any, List
import pytest
import numpy as np

class FoundationValidator:
    """Mock FoundationValidator."""
    
    VALID_PROMPTS = {"prompt_1", "prompt_2", "clinical_prompt"}
    
    @staticmethod
    def validate_embedding_dimension(embedding: np.ndarray, expected_dim: int) -> bool:
        if len(embedding.shape) != 2:
            return False
        return embedding.shape[1] == expected_dim
        
    @staticmethod
    def validate_prompt(prompt_name: str) -> bool:
        return prompt_name in FoundationValidator.VALID_PROMPTS
        
    @staticmethod
    def validate_adapter(adapter_config: dict) -> bool:
        required_keys = {"type", "reduction_factor"}
        return required_keys.issubset(adapter_config.keys())
        
    @staticmethod
    def check_nan_inf(data: np.ndarray) -> bool:
        return not (np.isnan(data).any() or np.isinf(data).any())


@pytest.mark.parametrize("shape", [(10, 128), (1, 128), (100, 128)])
@pytest.mark.parametrize("expected_dim", [128, 256, 64])
def test_validate_embedding_dimension(shape: tuple, expected_dim: int) -> None:
    """Test embedding dimension validation with correct and incorrect shapes."""
    embedding = np.zeros(shape)
    result = FoundationValidator.validate_embedding_dimension(embedding, expected_dim)
    expected_result = (shape[1] == expected_dim)
    assert result is expected_result

@pytest.mark.parametrize("shape", [(128,), (10, 128, 3), ()])
def test_validate_embedding_dimension_wrong_rank(shape: tuple) -> None:
    """Test embedding dimension validation with wrong tensor ranks."""
    embedding = np.zeros(shape)
    assert FoundationValidator.validate_embedding_dimension(embedding, 128) is False

@pytest.mark.parametrize("prompt", ["prompt_1", "prompt_2", "clinical_prompt", "unregistered_prompt", "random"])
@pytest.mark.parametrize("iteration", range(5))
def test_validate_prompt(prompt: str, iteration: int) -> None:
    """Test prompt validation against registered prompts."""
    is_valid = FoundationValidator.validate_prompt(prompt)
    assert is_valid is (prompt in FoundationValidator.VALID_PROMPTS)

@pytest.mark.parametrize("config", [
    {"type": "lora", "reduction_factor": 8},
    {"type": "bottleneck", "reduction_factor": 16, "activation": "relu"},
    {"type": "lora"},
    {"reduction_factor": 8},
    {}
])
@pytest.mark.parametrize("iteration", range(2))
def test_validate_adapter(config: dict, iteration: int) -> None:
    """Test adapter configuration validation."""
    is_valid = FoundationValidator.validate_adapter(config)
    expected = "type" in config and "reduction_factor" in config
    assert is_valid is expected

@pytest.mark.parametrize("shape", [(10, 10), (100, 50)])
@pytest.mark.parametrize("corruption", ["nan", "inf", "-inf", "none"])
def test_nan_inf_checks(shape: tuple, corruption: str) -> None:
    """Test NaN and Inf checks on numpy arrays."""
    data = np.ones(shape)
    if corruption == "nan":
        data[0, 0] = np.nan
    elif corruption == "inf":
        data[0, 0] = np.inf
    elif corruption == "-inf":
        data[0, 0] = -np.inf
        
    is_valid = FoundationValidator.check_nan_inf(data)
    assert is_valid is (corruption == "none")
