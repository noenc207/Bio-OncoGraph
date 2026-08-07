import numpy as np
import pytest
from bio_oncograph.ai.model.heads import ClassificationHead, SurvivalHead

def test_classification_head_init() -> None:
    """Test ClassificationHead init."""
    head = ClassificationHead(input_dim=128, num_classes=2)
    assert head.input_dim == 128
    assert head.num_classes == 2

def test_classification_head_forward() -> None:
    """Test ClassificationHead forward pass returns logits."""
    head = ClassificationHead(input_dim=128, num_classes=3)
    features = np.random.rand(5, 128)
    logits = head.forward(features)
    
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (5, 3)

def test_classification_head_output_shape() -> None:
    """Test ClassificationHead output shape exactly matches classes."""
    num_classes = 5
    batch_size = 10
    head = ClassificationHead(input_dim=64, num_classes=num_classes)
    features = np.random.rand(batch_size, 64)
    logits = head.forward(features)
    
    assert logits.shape == (batch_size, num_classes)

def test_survival_head_init() -> None:
    """Test SurvivalHead init."""
    head = SurvivalHead(input_dim=128)
    assert head.input_dim == 128

def test_survival_head_forward() -> None:
    """Test SurvivalHead forward pass returns risk scores."""
    head = SurvivalHead(input_dim=128)
    features = np.random.rand(4, 128)
    risk = head.forward(features)
    
    assert isinstance(risk, np.ndarray)
    assert risk.shape == (4, 1)

def test_survival_head_output_shape() -> None:
    """Test SurvivalHead output shape."""
    batch_size = 8
    head = SurvivalHead(input_dim=64)
    features = np.random.rand(batch_size, 64)
    risk = head.forward(features)
    
    assert risk.shape == (batch_size, 1)

def test_classification_head_invalid_input() -> None:
    """Test ClassificationHead raises error on invalid input shape."""
    head = ClassificationHead(input_dim=128, num_classes=2)
    features = np.random.rand(4, 64)  # Mismatch input_dim
    
    with pytest.raises((ValueError, RuntimeError, Exception)):
        head.forward(features)

def test_survival_head_invalid_input() -> None:
    """Test SurvivalHead raises error on invalid input shape."""
    head = SurvivalHead(input_dim=64)
    features = np.random.rand(4, 32)  # Mismatch input_dim
    
    with pytest.raises((ValueError, RuntimeError, Exception)):
        head.forward(features)
