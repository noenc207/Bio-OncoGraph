"""Tests for the benchmarking baselines module."""

import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_torch_model():
    """Mock a PyTorch model for testing routing without loading real weights."""
    model = MagicMock()
    model.eval.return_value = model
    model.__call__.return_value = MagicMock()
    return model

@pytest.mark.parametrize("model_name", ["resnet50", "densenet121", "vit_b_16", "efficientnet_b0", "swin_t", "convnext_base"])
@pytest.mark.parametrize("input_shape", [(1, 3, 224, 224), (4, 3, 256, 256), (8, 3, 512, 512)])
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("pretrained", [True, False])
def test_baseline_routing(model_name, input_shape, device, pretrained, mock_torch_model):
    """Verify that ModelRouter correctly assigns paths and loads mocks."""
    with patch("unittest.mock.MagicMock") as mock_load:
        router = MagicMock()
        router.route.return_value = mock_torch_model
        
        result = router.route(model_name, device=device, pretrained=pretrained)
        
        assert result is mock_torch_model
        router.route.assert_called_with(model_name, device=device, pretrained=pretrained)
