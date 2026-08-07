"""Tests for Dataset Metadata functionality."""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import hashlib

@dataclass
class Metadata:
    """Dataclass representing dataset metadata."""
    dataset_id: str
    version: str
    description: str
    num_samples: int
    features: List[str]
    checksum: str
    tags: Dict[str, str] = field(default_factory=dict)

    def validate_checksum(self, data: bytes) -> bool:
        """Validate the checksum against provided data."""
        return hashlib.sha256(data).hexdigest() == self.checksum

    def __str__(self) -> str:
        return f"Metadata({self.dataset_id} v{self.version}: {self.num_samples} samples)"

@pytest.fixture
def sample_metadata():
    """Provide a sample Metadata instance for testing."""
    return Metadata(
        dataset_id="ds_001",
        version="1.0.0",
        description="Sample dataset",
        num_samples=1000,
        features=["feat1", "feat2"],
        checksum=hashlib.sha256(b"test data").hexdigest(),
        tags={"domain": "oncology"}
    )

def test_metadata_initialization(sample_metadata):
    """Test successful initialization of Metadata."""
    assert sample_metadata.dataset_id == "ds_001"
    assert sample_metadata.version == "1.0.0"
    assert sample_metadata.num_samples == 1000
    assert len(sample_metadata.features) == 2

def test_metadata_default_tags():
    """Test that tags default to an empty dict if not provided."""
    meta = Metadata(
        dataset_id="ds_002",
        version="0.1",
        description="Desc",
        num_samples=10,
        features=["a"],
        checksum="abc"
    )
    assert meta.tags == {}

def test_validate_checksum_success(sample_metadata):
    """Test checksum validation with correct data."""
    assert sample_metadata.validate_checksum(b"test data") is True

def test_validate_checksum_failure(sample_metadata):
    """Test checksum validation with incorrect data."""
    assert sample_metadata.validate_checksum(b"wrong data") is False

def test_metadata_string_representation(sample_metadata):
    """Test the __str__ method of Metadata."""
    assert str(sample_metadata) == "Metadata(ds_001 v1.0.0: 1000 samples)"

def test_metadata_immutability_of_features(sample_metadata):
    """Test modifying features list directly."""
    sample_metadata.features.append("feat3")
    assert "feat3" in sample_metadata.features

def test_metadata_missing_required_fields():
    """Test initialization fails when required fields are missing."""
    with pytest.raises(TypeError):
        Metadata(dataset_id="ds_001") # Missing version, description, etc.

def test_metadata_empty_features():
    """Test Metadata with empty features list."""
    meta = Metadata("id", "v1", "desc", 0, [], "hash")
    assert meta.features == []

def test_metadata_invalid_checksum_type():
    """Test checksum as string behaves correctly."""
    meta = Metadata("id", "v1", "desc", 0, [], "invalid_hash")
    assert meta.validate_checksum(b"data") is False

def test_metadata_large_num_samples():
    """Test Metadata with a large number of samples."""
    meta = Metadata("id", "v1", "desc", 1_000_000_000, [], "hash")
    assert meta.num_samples == 1_000_000_000

def test_metadata_update_tags(sample_metadata):
    """Test updating tags dictionary."""
    sample_metadata.tags["new_tag"] = "value"
    assert sample_metadata.tags["new_tag"] == "value"

def test_metadata_multiple_instances():
    """Test that multiple instances don't share default mutable state."""
    meta1 = Metadata("id1", "v1", "desc", 0, [], "hash")
    meta2 = Metadata("id2", "v1", "desc", 0, [], "hash")
    meta1.tags["a"] = "b"
    assert meta2.tags == {}
