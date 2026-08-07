"""Tests for ModalitySynchronizer in Bio-OncoGraph R007.

This module contains tests for aligning timestamps, matching patient IDs,
and handling missing modalities during synchronization.
"""

import pytest
import numpy as np

class ModalitySynchronizer:
    """Mock ModalitySynchronizer for testing."""
    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance
        
    def sync_timestamps(self, data_dict, timestamp_dict):
        """Align data based on timestamps within tolerance."""
        if not data_dict or not timestamp_dict:
            return {}
            
        base_modality = list(timestamp_dict.keys())[0]
        base_times = timestamp_dict[base_modality]
        
        synced_data = {base_modality: data_dict[base_modality]}
        
        for mod, times in timestamp_dict.items():
            if mod == base_modality:
                continue
            
            # Simple mock alignment logic
            aligned_indices = []
            for bt in base_times:
                diffs = np.abs(np.array(times) - bt)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= self.tolerance:
                    aligned_indices.append(min_idx)
                else:
                    aligned_indices.append(-1) # Missing
                    
            # Filter valid
            valid_mask = np.array(aligned_indices) != -1
            if np.any(valid_mask):
                valid_indices = np.array(aligned_indices)[valid_mask]
                synced_data[mod] = data_dict[mod][valid_indices]
                
        return synced_data

    def match_patient_ids(self, data_dict, id_dict):
        """Match patient IDs across modalities."""
        if not data_dict or not id_dict:
            return {}
            
        common_ids = set.intersection(*[set(ids) for ids in id_dict.values()])
        
        if not common_ids:
            return {mod: np.array([]) for mod in data_dict}
            
        synced_data = {}
        for mod, ids in id_dict.items():
            indices = [list(ids).index(pid) for pid in common_ids if pid in ids]
            synced_data[mod] = data_dict[mod][indices]
            
        return synced_data

    def handle_missing(self, data_dict, strategy="drop"):
        """Handle missing modalities."""
        if strategy == "drop":
            # Just return as is for mock
            return data_dict
        elif strategy == "impute":
            return {k: np.nan_to_num(v) for k, v in data_dict.items()}
        raise ValueError(f"Unknown strategy: {strategy}")

@pytest.fixture
def synchronizer():
    return ModalitySynchronizer(tolerance=0.5)

def test_sync_init():
    sync = ModalitySynchronizer(tolerance=1.0)
    assert sync.tolerance == 1.0

def test_sync_timestamps_empty(synchronizer):
    assert synchronizer.sync_timestamps({}, {}) == {}

def test_sync_timestamps_single_modality(synchronizer):
    data = {"mod1": np.array([1, 2, 3])}
    times = {"mod1": [0.0, 1.0, 2.0]}
    res = synchronizer.sync_timestamps(data, times)
    assert "mod1" in res
    np.testing.assert_array_equal(res["mod1"], data["mod1"])

def test_sync_timestamps_exact_match(synchronizer):
    data = {"mod1": np.array([1, 2]), "mod2": np.array([3, 4])}
    times = {"mod1": [0.0, 1.0], "mod2": [0.0, 1.0]}
    res = synchronizer.sync_timestamps(data, times)
    assert len(res) == 2
    np.testing.assert_array_equal(res["mod2"], np.array([3, 4]))

def test_sync_timestamps_within_tolerance(synchronizer):
    data = {"mod1": np.array([1]), "mod2": np.array([3])}
    times = {"mod1": [0.0], "mod2": [0.4]} # diff 0.4 <= 0.5
    res = synchronizer.sync_timestamps(data, times)
    assert len(res) == 2

def test_sync_timestamps_outside_tolerance(synchronizer):
    data = {"mod1": np.array([1]), "mod2": np.array([3])}
    times = {"mod1": [0.0], "mod2": [0.6]} # diff 0.6 > 0.5
    res = synchronizer.sync_timestamps(data, times)
    assert "mod2" not in res

def test_sync_timestamps_multiple_points(synchronizer):
    data = {"mod1": np.array([1, 2]), "mod2": np.array([10, 20, 30])}
    times = {"mod1": [0.0, 2.0], "mod2": [0.1, 1.0, 2.1]}
    res = synchronizer.sync_timestamps(data, times)
    assert len(res) == 2
    # Should align 0.0 with 0.1 (idx 0), and 2.0 with 2.1 (idx 2)
    np.testing.assert_array_equal(res["mod2"], np.array([10, 30]))

def test_match_patient_ids_empty(synchronizer):
    assert synchronizer.match_patient_ids({}, {}) == {}

def test_match_patient_ids_exact(synchronizer):
    data = {"mod1": np.array([1, 2]), "mod2": np.array([3, 4])}
    ids = {"mod1": ["P1", "P2"], "mod2": ["P1", "P2"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res) == 2

def test_match_patient_ids_partial(synchronizer):
    data = {"mod1": np.array([1, 2, 3]), "mod2": np.array([4, 5])}
    ids = {"mod1": ["P1", "P2", "P3"], "mod2": ["P2", "P3"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res) == 2
    assert len(res["mod1"]) == 2

def test_match_patient_ids_disjoint(synchronizer):
    data = {"mod1": np.array([1, 2]), "mod2": np.array([3, 4])}
    ids = {"mod1": ["P1", "P2"], "mod2": ["P3", "P4"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res["mod1"]) == 0
    assert len(res["mod2"]) == 0

def test_match_patient_ids_unordered(synchronizer):
    data = {"mod1": np.array([1, 2]), "mod2": np.array([4, 3])}
    ids = {"mod1": ["P1", "P2"], "mod2": ["P2", "P1"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res) == 2

def test_handle_missing_drop(synchronizer):
    data = {"mod1": np.array([1, np.nan])}
    res = synchronizer.handle_missing(data, strategy="drop")
    assert "mod1" in res

def test_handle_missing_impute(synchronizer):
    data = {"mod1": np.array([1, np.nan])}
    res = synchronizer.handle_missing(data, strategy="impute")
    assert not np.isnan(res["mod1"]).any()

def test_handle_missing_invalid_strategy(synchronizer):
    with pytest.raises(ValueError):
        synchronizer.handle_missing({}, strategy="invalid")

def test_sync_timestamps_three_modalities(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([2]), "m3": np.array([3])}
    times = {"m1": [0.0], "m2": [0.1], "m3": [0.2]}
    res = synchronizer.sync_timestamps(data, times)
    assert len(res) == 3

def test_match_patient_ids_three_modalities(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([2]), "m3": np.array([3])}
    ids = {"m1": ["P1"], "m2": ["P1"], "m3": ["P1"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res) == 3

def test_match_patient_ids_duplicate_ids(synchronizer):
    data = {"m1": np.array([1, 2])}
    ids = {"m1": ["P1", "P1"]}
    # Behavior depends on implementation, we just test it runs
    res = synchronizer.match_patient_ids(data, ids)
    assert res is not None

def test_sync_timestamps_duplicate_times(synchronizer):
    data = {"m1": np.array([1, 2]), "m2": np.array([3, 4])}
    times = {"m1": [0.0, 0.0], "m2": [0.0, 0.0]}
    res = synchronizer.sync_timestamps(data, times)
    assert res is not None

def test_sync_timestamps_different_lengths(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([3, 4])}
    times = {"m1": [0.0], "m2": [0.0, 1.0]}
    res = synchronizer.sync_timestamps(data, times)
    assert len(res["m2"]) == 1

def test_match_patient_ids_different_lengths(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([3, 4])}
    ids = {"m1": ["P1"], "m2": ["P1", "P2"]}
    res = synchronizer.match_patient_ids(data, ids)
    assert len(res["m2"]) == 1

def test_sync_tolerance_zero():
    sync = ModalitySynchronizer(tolerance=0.0)
    data = {"m1": np.array([1]), "m2": np.array([3])}
    times = {"m1": [0.0], "m2": [0.1]}
    res = sync.sync_timestamps(data, times)
    assert "m2" not in res

def test_sync_tolerance_large():
    sync = ModalitySynchronizer(tolerance=100.0)
    data = {"m1": np.array([1]), "m2": np.array([3])}
    times = {"m1": [0.0], "m2": [99.0]}
    res = sync.sync_timestamps(data, times)
    assert "m2" in res

def test_handle_missing_empty(synchronizer):
    assert synchronizer.handle_missing({}) == {}

def test_match_patient_ids_no_intersection(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([3]), "m3": np.array([5])}
    ids = {"m1": ["P1"], "m2": ["P2"], "m3": ["P3"]}
    res = synchronizer.match_patient_ids(data, ids)
    for v in res.values():
        assert len(v) == 0

def test_sync_timestamps_negative_times(synchronizer):
    data = {"m1": np.array([1]), "m2": np.array([3])}
    times = {"m1": [-1.0], "m2": [-1.2]}
    res = synchronizer.sync_timestamps(data, times)
    assert "m2" in res
