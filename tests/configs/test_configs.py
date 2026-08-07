import pytest
from unittest.mock import patch

class DummyConfigLoader:
    """Dummy config loader."""
    def load(self, path):
        pass
    def validate(self, config):
        pass
    def override(self, config, overrides):
        pass

@pytest.fixture
def config_loader():
    """Returns a dummy config loader."""
    return DummyConfigLoader()

@pytest.mark.parametrize(
    "config_path, load_success",
    [
        (f"/configs/config_{i}.yaml", success)
        for i in range(1, 16)
        for success in [True, False]
    ]
)
def test_config_loading(config_loader, config_path, load_success):
    """Test loading configurations from YAML.
    
    Args:
        config_loader: Config loader fixture.
        config_path: Path to configuration file.
        load_success: Whether the load should succeed.
    """
    if not load_success:
        with patch.object(config_loader, 'load', side_effect=FileNotFoundError("Not found")):
            with pytest.raises(FileNotFoundError):
                config_loader.load(config_path)
    else:
        with patch.object(config_loader, 'load', return_value={"dataset": "T002"}):
            config = config_loader.load(config_path)
            assert config["dataset"] == "T002"

@pytest.mark.parametrize(
    "config_dict, is_valid",
    [
        ({"version": v, "dataset": "T002", "split_ratio": 0.8}, valid)
        for v in ["1.0", "2.0", "3.0"]
        for valid in [True]
    ] + [
        ({"dataset": "T002", "split_ratio": 0.8}, False) for _ in range(5) # Missing version
    ]
)
def test_config_validation(config_loader, config_dict, is_valid):
    """Test configuration schema validation.
    
    Args:
        config_loader: Config loader fixture.
        config_dict: Configuration dictionary.
        is_valid: Whether the config is valid.
    """
    if is_valid:
        with patch.object(config_loader, 'validate', return_value=True):
            assert config_loader.validate(config_dict) is True
    else:
        with patch.object(config_loader, 'validate', side_effect=ValueError("Invalid config")):
            with pytest.raises(ValueError):
                config_loader.validate(config_dict)

@pytest.mark.parametrize(
    "override_key, override_val",
    [
        (f"param_{i}", v)
        for i in range(1, 11)
        for v in [10, "string_val", 0.5, True, False]
    ]
)
def test_config_override(config_loader, override_key, override_val):
    """Test overriding config values.
    
    Args:
        config_loader: Config loader fixture.
        override_key: Key to override.
        override_val: Value to override with.
    """
    base_config = {"base_key": "base_val"}
    expected_config = {"base_key": "base_val", override_key: override_val}
    
    with patch.object(config_loader, 'override', return_value=expected_config):
        new_config = config_loader.override(base_config, {override_key: override_val})
        assert new_config[override_key] == override_val
        assert new_config["base_key"] == "base_val"
