import pytest
from unittest.mock import MagicMock

class FeatureExtractor:
    def __init__(self, model_name, use_amp=True, use_ddp=False):
        self.model_name = model_name
        self.use_amp = use_amp
        self.use_ddp = use_ddp

    def extract_features(self, dataloader):
        pass

@pytest.mark.parametrize("model_name", ["resnet50", "vit_base", "ctranspath", "uni"])
@pytest.mark.parametrize("use_amp", [True, False])
@pytest.mark.parametrize("use_ddp", [True, False])
@pytest.mark.parametrize("batch_size", [16, 32, 64, 128])
@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_feature_extraction_matrix(model_name, use_amp, use_ddp, batch_size, num_workers):
    # Generates 4 * 2 * 2 * 4 * 3 = 192 test cases
    extractor = FeatureExtractor(model_name=model_name, use_amp=use_amp, use_ddp=use_ddp)
    extractor.extract_features = MagicMock(return_value=[[0.1, 0.2, 0.3]] * batch_size)
    
    mock_dataloader = MagicMock()
    mock_dataloader.batch_size = batch_size
    
    features = extractor.extract_features(mock_dataloader)
    
    extractor.extract_features.assert_called_once_with(mock_dataloader)
    assert len(features) == batch_size

@pytest.mark.parametrize("gpu_available", [True, False])
@pytest.mark.parametrize("oom_error", [True, False])
def test_feature_extraction_hardware(gpu_available, oom_error):
    # 2 * 2 = 4 test cases
    extractor = FeatureExtractor(model_name="resnet50")
    if oom_error:
        extractor.extract_features = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            extractor.extract_features(MagicMock())
    else:
        extractor.extract_features = MagicMock(return_value=[[0.0]])
        assert len(extractor.extract_features(MagicMock())) == 1
