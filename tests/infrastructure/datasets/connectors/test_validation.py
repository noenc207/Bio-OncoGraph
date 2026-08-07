"""Tests for Dataset Validation functionality."""

import pytest
from typing import List, Dict, Any

class Validator:
    """Validates dataset integrity."""
    def __init__(self, expected_files: List[str], expected_splits: List[str]):
        self.expected_files = expected_files
        self.expected_splits = expected_splits

    def validate_files(self, existing_files: List[str]) -> bool:
        """Check for missing or duplicate files."""
        if len(existing_files) != len(set(existing_files)):
            raise ValueError("Duplicate files found")
        for f in self.expected_files:
            if f not in existing_files:
                raise FileNotFoundError(f"Missing file: {f}")
        return True

    def validate_splits(self, splits: Dict[str, Any]) -> bool:
        """Check if all required splits are present and valid."""
        for split in self.expected_splits:
            if split not in splits:
                raise ValueError(f"Missing split: {split}")
            if not splits[split]:
                raise ValueError(f"Empty split: {split}")
        return True

@pytest.fixture
def validator():
    return Validator(
        expected_files=["data.csv", "meta.json"],
        expected_splits=["train", "test", "val"]
    )

def test_validate_files_success(validator):
    """Test successful validation of files."""
    assert validator.validate_files(["data.csv", "meta.json", "extra.txt"]) is True

def test_validate_files_missing(validator):
    """Test validation fails when files are missing."""
    with pytest.raises(FileNotFoundError, match="Missing file: meta.json"):
        validator.validate_files(["data.csv"])

def test_validate_files_duplicates(validator):
    """Test validation fails on duplicate files."""
    with pytest.raises(ValueError, match="Duplicate files found"):
        validator.validate_files(["data.csv", "data.csv", "meta.json"])

def test_validate_splits_success(validator):
    """Test successful validation of splits."""
    splits = {"train": [1,2], "test": [3], "val": [4]}
    assert validator.validate_splits(splits) is True

def test_validate_splits_missing(validator):
    """Test validation fails when a split is missing."""
    splits = {"train": [1,2], "test": [3]}
    with pytest.raises(ValueError, match="Missing split: val"):
        validator.validate_splits(splits)

def test_validate_splits_empty(validator):
    """Test validation fails when a split is empty."""
    splits = {"train": [1,2], "test": [], "val": [4]}
    with pytest.raises(ValueError, match="Empty split: test"):
        validator.validate_splits(splits)

def test_validate_no_expected_files():
    """Test validation with empty expected files list."""
    val = Validator(expected_files=[], expected_splits=["train"])
    assert val.validate_files(["anything.txt"]) is True

def test_validate_no_expected_splits():
    """Test validation with empty expected splits list."""
    val = Validator(expected_files=["data.csv"], expected_splits=[])
    assert val.validate_splits({}) is True

def test_validate_exact_match(validator):
    """Test validation with exact matches."""
    assert validator.validate_files(["data.csv", "meta.json"]) is True

def test_validate_files_empty_input(validator):
    """Test validation with empty input list."""
    with pytest.raises(FileNotFoundError):
        validator.validate_files([])

def test_validate_splits_extra_splits(validator):
    """Test validation succeeds when extra splits are provided."""
    splits = {"train": [1], "test": [2], "val": [3], "extra": [4]}
    assert validator.validate_splits(splits) is True

def test_validator_initialization():
    """Test Validator initialization."""
    v = Validator(["a"], ["b"])
    assert v.expected_files == ["a"]
    assert v.expected_splits == ["b"]
