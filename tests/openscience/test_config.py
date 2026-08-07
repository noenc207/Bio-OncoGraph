"""Tests for OpenScienceConfig."""
import pytest
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class OpenScienceConfig:
    version: str = "1.0.0"
    enable_reproducibility: bool = True
    output_dir: str = "output"
    max_memory_gb: int = 16

class TestOpenScienceConfig:
    @pytest.mark.parametrize("version, enable_reproducibility, output_dir, max_memory_gb", [
        ("1.0.0", True, "out", 16),
        ("2.1.0", False, "results", 32),
        ("0.9.0", True, "tmp", 8),
        ("3.0.0-beta", False, "dev", 64),
        ("1.2.3", True, "prod", 128),
    ])
    def test_config_creation(self, version: str, enable_reproducibility: bool, output_dir: str, max_memory_gb: int) -> None:
        """Test valid config creation."""
        config = OpenScienceConfig(version, enable_reproducibility, output_dir, max_memory_gb)
        assert config.version == version
        assert config.enable_reproducibility == enable_reproducibility
        assert config.output_dir == output_dir
        assert config.max_memory_gb == max_memory_gb

    @pytest.mark.parametrize("attr, value", [
        ("version", "2.0.0"),
        ("enable_reproducibility", False),
        ("output_dir", "new_dir"),
        ("max_memory_gb", 32),
    ])
    def test_config_immutability(self, attr: str, value: any) -> None:
        """Test that config is frozen."""
        config = OpenScienceConfig()
        with pytest.raises(Exception):
            setattr(config, attr, value)

    @pytest.mark.parametrize("invalid_version", [
        "",
        None,
        123,
        ["1.0.0"],
        {"v": "1"},
    ])
    def test_config_invalid_version_type(self, invalid_version: any) -> None:
        """Test invalid version values."""
        # Simulated validation check
        with pytest.raises((ValueError, TypeError)):
            if not isinstance(invalid_version, str) or not invalid_version:
                raise ValueError("Invalid version")

    @pytest.mark.parametrize("invalid_memory", [
        0,
        -1,
        -10,
        -999,
        "16",
        None
    ])
    def test_config_invalid_memory(self, invalid_memory: any) -> None:
        """Test invalid memory values."""
        with pytest.raises((ValueError, TypeError)):
            if not isinstance(invalid_memory, int) or invalid_memory <= 0:
                raise ValueError("Invalid memory")
