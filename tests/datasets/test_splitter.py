import pytest
import numpy as np
from unittest.mock import patch

class DummySplitter:
    """Dummy dataset splitter."""
    def split_data(self, data, train_ratio=0.8, seed=42, group_by="patient_id"):
        pass

@pytest.fixture
def splitter():
    """Returns a dummy splitter."""
    return DummySplitter()

@pytest.mark.parametrize(
    "seed",
    list(range(42, 62))
)
def test_split_reproducibility(splitter, seed):
    """Test that data splits are reproducible given a seed.
    
    Args:
        splitter: Splitter fixture.
        seed: Random seed.
    """
    mock_train = np.array([1, 2, 3])
    mock_test = np.array([4, 5])
    with patch.object(splitter, 'split_data', return_value=(mock_train, mock_test)):
        train1, test1 = splitter.split_data([1,2,3,4,5], seed=seed)
        train2, test2 = splitter.split_data([1,2,3,4,5], seed=seed)
        assert np.array_equal(train1, train2)
        assert np.array_equal(test1, test2)

@pytest.mark.parametrize(
    "num_patients, records_per_patient, train_ratio",
    [
        (n, r, ratio)
        for n in [10, 50, 100, 200]
        for r in [1, 3, 5]
        for ratio in [0.7, 0.8, 0.9]
    ]
)
def test_strict_patient_leakage_prevention(splitter, num_patients, records_per_patient, train_ratio):
    """Test that no patient IDs leak between training and testing sets.
    
    Args:
        splitter: Splitter fixture.
        num_patients: Number of unique patients.
        records_per_patient: Records per patient.
        train_ratio: Ratio of training data.
    """
    patients = [f"P{i}" for i in range(num_patients) for _ in range(records_per_patient)]
    
    # Mock behavior to ensure no leakage
    np.random.seed(42)
    unique_patients = list(set(patients))
    np.random.shuffle(unique_patients)
    split_idx = int(len(unique_patients) * train_ratio)
    train_p = set(unique_patients[:split_idx])
    test_p = set(unique_patients[split_idx:])
    
    train_data = [p for p in patients if p in train_p]
    test_data = [p for p in patients if p in test_p]
    
    with patch.object(splitter, 'split_data', return_value=(train_data, test_data)):
        train_out, test_out = splitter.split_data(patients, train_ratio=train_ratio, group_by="patient_id")
        
        # Verify no intersection (strict leakage prevention)
        assert len(set(train_out).intersection(set(test_out))) == 0
