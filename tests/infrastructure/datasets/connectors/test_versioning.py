"""Tests for Dataset Versioning."""

import pytest
from typing import List, Dict

class VersionTracker:
    """Tracks different versions of datasets."""
    def __init__(self):
        self._versions: Dict[str, List[str]] = {}
        self._current: Dict[str, str] = {}

    def add_version(self, dataset_id: str, version: str) -> None:
        """Add a new version for a dataset."""
        if dataset_id not in self._versions:
            self._versions[dataset_id] = []
        if version in self._versions[dataset_id]:
            raise ValueError(f"Version {version} already exists for {dataset_id}")
        self._versions[dataset_id].append(version)
        self._current[dataset_id] = version

    def get_current_version(self, dataset_id: str) -> str:
        """Get the latest added version for a dataset."""
        if dataset_id not in self._current:
            raise KeyError(f"No versions found for {dataset_id}")
        return self._current[dataset_id]

    def get_all_versions(self, dataset_id: str) -> List[str]:
        """List all versions for a dataset."""
        return self._versions.get(dataset_id, [])

    def set_active_version(self, dataset_id: str, version: str) -> None:
        """Set a specific version as the active/current one."""
        if dataset_id not in self._versions or version not in self._versions[dataset_id]:
            raise ValueError(f"Version {version} not found for {dataset_id}")
        self._current[dataset_id] = version

@pytest.fixture
def tracker():
    return VersionTracker()

def test_add_version(tracker):
    """Test adding a new version."""
    tracker.add_version("ds1", "1.0")
    assert tracker.get_all_versions("ds1") == ["1.0"]

def test_get_current_version(tracker):
    """Test retrieving current version."""
    tracker.add_version("ds1", "1.0")
    tracker.add_version("ds1", "1.1")
    assert tracker.get_current_version("ds1") == "1.1"

def test_get_current_version_missing(tracker):
    """Test getting current version for unknown dataset."""
    with pytest.raises(KeyError, match="No versions found"):
        tracker.get_current_version("unknown")

def test_add_duplicate_version(tracker):
    """Test adding a duplicate version raises ValueError."""
    tracker.add_version("ds1", "1.0")
    with pytest.raises(ValueError, match="already exists"):
        tracker.add_version("ds1", "1.0")

def test_get_all_versions_missing(tracker):
    """Test getting all versions for unknown dataset returns empty."""
    assert tracker.get_all_versions("unknown") == []

def test_set_active_version(tracker):
    """Test setting an older version as active."""
    tracker.add_version("ds1", "1.0")
    tracker.add_version("ds1", "1.1")
    tracker.set_active_version("ds1", "1.0")
    assert tracker.get_current_version("ds1") == "1.0"

def test_set_active_version_invalid(tracker):
    """Test setting active version to a non-existent version."""
    tracker.add_version("ds1", "1.0")
    with pytest.raises(ValueError, match="not found"):
        tracker.set_active_version("ds1", "2.0")

def test_multiple_datasets(tracker):
    """Test tracking versions for multiple datasets independently."""
    tracker.add_version("ds1", "1.0")
    tracker.add_version("ds2", "2.0")
    assert tracker.get_current_version("ds1") == "1.0"
    assert tracker.get_current_version("ds2") == "2.0"

def test_version_order_preserved(tracker):
    """Test that versions are listed in order of addition."""
    versions = ["1.0", "0.9", "1.1-beta"]
    for v in versions:
        tracker.add_version("ds1", v)
    assert tracker.get_all_versions("ds1") == versions

def test_set_active_missing_dataset(tracker):
    """Test setting active version on unknown dataset."""
    with pytest.raises(ValueError, match="not found"):
        tracker.set_active_version("unknown", "1.0")

def test_initial_state(tracker):
    """Test tracker initialization state."""
    assert tracker._versions == {}
    assert tracker._current == {}

def test_add_version_updates_current(tracker):
    """Test adding version implicitly sets it as current."""
    tracker.add_version("ds", "v1")
    assert tracker.get_current_version("ds") == "v1"
    tracker.add_version("ds", "v2")
    assert tracker.get_current_version("ds") == "v2"
